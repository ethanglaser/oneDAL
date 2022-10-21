/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel_brute_force_impl.hpp"

namespace oneapi::dal::knn::backend {

using idx_t = std::int32_t;

using dal::backend::context_gpu;

template <typename Task>
using descriptor_t = detail::descriptor_base<Task>;

using voting_t = ::oneapi::dal::knn::voting_mode;

using daal_distance_t = daal::algorithms::internal::PairwiseDistanceType;

template <typename T1, typename T2>
sycl::event copy_with_sqrt(sycl::queue& q,
                           const pr::ndview<T2, 2>& src,
                           pr::ndview<T1, 2>& dst,
                           const bk::event_vector& deps = {}) {
    static_assert(de::is_floating_point<T1>());
    static_assert(de::is_floating_point<T2>());
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    const pr::ndshape<2> dst_shape = dst.get_shape();
    ONEDAL_ASSERT(dst_shape == src.get_shape());
    T1* const dst_ptr = dst.get_mutable_data();
    const T2* const src_ptr = src.get_data();
    const auto dst_stride = dst.get_leading_stride();
    const auto src_stride = src.get_leading_stride();
    const auto cp_range = bk::make_range_2d(dst_shape[0], dst_shape[1]);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(cp_range, [=](sycl::id<2> idx) {
            T1& dst_ref = *(dst_ptr + idx[0] * dst_stride + idx[1]);
            const T2& val_ref = *(src_ptr + idx[0] * src_stride + idx[1]);
            dst_ref = sycl::sqrt(val_ref);
        });
    });
}

inline bool is_col_major(const table& t) {
    const auto t_layout = t.get_data_layout();
    return t_layout == decltype(t_layout)::column_major;
}

template <typename Float, bool is_cm>
struct ndarray_t_map;

template <typename Float>
struct ndarray_t_map<Float, true> {
    using type = pr::ndarray<Float, 2, pr::ndorder::f>;
};

template <typename Float>
struct ndarray_t_map<Float, false> {
    using type = pr::ndarray<Float, 2, pr::ndorder::c>;
};

template <typename Float, bool is_cm>
using ndarray_t = typename ndarray_t_map<Float, is_cm>::type;

template <typename Type, pr::ndorder order>
constexpr pr::ndorder get_ndorder(const pr::ndarray<Type, 2, order>&) {
    return order;
}

template <typename Float, typename Task, bool cm_train, bool cm_query>
static infer_result<Task> kernel(const context_gpu& ctx,
                                 const descriptor_t<Task>& desc,
                                 const table& infer,
                                 const model<Task>& m) {
    using res_t = response_t<Task>;

    auto distance_impl = detail::get_distance_impl(desc);
    if (!distance_impl) {
        throw internal_error{ de::error_messages::unknown_distance_type() };
    }

    const bool is_minkowski_distance =
        distance_impl->get_daal_distance_type() == daal_distance_t::minkowski;
    const bool is_chebyshev_distance =
        distance_impl->get_daal_distance_type() == daal_distance_t::chebyshev;
    const bool is_cosine_distance =
        distance_impl->get_daal_distance_type() == daal_distance_t::cosine;
    const bool is_euclidean_distance =
        is_minkowski_distance && (distance_impl->get_degree() == 2.0);

    const auto trained_model = dynamic_cast_to_knn_model<Task, brute_force_model_impl<Task>>(m);
    const auto train = trained_model->get_data();
    const auto resps = trained_model->get_responses();

    const std::int64_t infer_row_count = infer.get_row_count();
    const std::int64_t feature_count = train.get_column_count();

    const std::int64_t class_count = desc.get_class_count();
    const std::int64_t neighbor_count = desc.get_neighbor_count();

    ONEDAL_ASSERT(train.get_column_count() == infer.get_column_count());

    auto& queue = ctx.get_queue();
    auto& comm = ctx.get_communicator();

    bk::interop::execution_context_guard guard(queue);

    auto arr_responses = array<res_t>{};
    if (desc.get_result_options().test(result_options::responses)) {
        arr_responses = array<res_t>::empty(queue, infer_row_count, sycl::usm::alloc::device);
    }
    auto arr_distances = array<Float>{};
    if (desc.get_result_options().test(result_options::distances) ||
        (desc.get_voting_mode() == voting_t::distance)) {
        const auto length = de::check_mul_overflow(infer_row_count, neighbor_count);
        arr_distances = array<Float>::empty(queue, length, sycl::usm::alloc::device);
    }
    auto arr_indices = array<idx_t>{};
    if (desc.get_result_options().test(result_options::indices)) {
        const auto length = de::check_mul_overflow(infer_row_count, neighbor_count);
        arr_indices = array<idx_t>::empty(queue, length, sycl::usm::alloc::device);
    }

    using train_t = ndarray_t<Float, cm_train>;
    auto train_var = pr::table2ndarray_variant<Float>(queue, train, sycl::usm::alloc::device);
    train_t train_data = std::get<train_t>(train_var);

    using query_t = ndarray_t<Float, cm_query>;
    auto query_var = pr::table2ndarray_variant<Float>(queue, infer, sycl::usm::alloc::device);
    query_t query_data = std::get<query_t>(query_var);

    auto resps_data = desc.get_result_options().test(result_options::responses)
                          ? pr::table2ndarray_1d<res_t>(queue, resps, sycl::usm::alloc::device)
                          : pr::ndarray<res_t, 1>{};

    const std::int64_t infer_block = pr::propose_query_block<Float>(queue, feature_count);
    const std::int64_t train_block = pr::propose_train_block<Float>(queue, feature_count);

    knn_callback<Float, Task> callback(queue,
                                       comm,
                                       desc.get_result_options(),
                                       infer_block,
                                       infer_row_count,
                                       neighbor_count);

    callback.set_inp_responses(resps_data);
    callback.set_responses(arr_responses);
    callback.set_distances(arr_distances);
    callback.set_indices(arr_indices);

    if constexpr (std::is_same_v<Task, task::classification>) {
        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::uniform)) {
            callback.set_uniform_voting(
                std::move(pr::make_uniform_voting(queue, infer_block, neighbor_count)));
        }

        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::distance)) {
            callback.set_distance_voting(
                std::move(pr::make_distance_voting<Float>(queue, infer_block, class_count)));
        }
    }

    if constexpr (std::is_same_v<Task, task::regression>) {
        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::uniform)) {
            callback.set_uniform_regression(
                std::move(pr::make_uniform_regression<res_t>(queue, infer_block, neighbor_count)));
        }

        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::distance)) {
            callback.set_distance_regression(
                std::move(pr::make_distance_regression<Float>(queue, infer_block, neighbor_count)));
        }
    }

    if (is_cosine_distance) {
        using dst_t = pr::cosine_distance<Float>;
        [[maybe_unused]] constexpr auto order = get_ndorder(train_data);
        using search_t = pr::search_engine<Float, dst_t, order>;

        const dst_t dist{ queue };
        const search_t search{ queue, train_data, train_block, dist };
        search(query_data, callback, infer_block, neighbor_count).wait_and_throw();
    }

    if (is_chebyshev_distance) {
        using dst_t = pr::chebyshev_distance<Float>;
        [[maybe_unused]] constexpr auto order = get_ndorder(train_data);
        using search_t = pr::search_engine<Float, dst_t, order>;

        const dst_t dist{ queue };
        const search_t search{ queue, train_data, train_block, dist };
        search(query_data, callback, infer_block, neighbor_count).wait_and_throw();
    }

    if (is_euclidean_distance) {
        using dst_t = pr::squared_l2_distance<Float>;
        [[maybe_unused]] constexpr auto order = get_ndorder(train_data);
        using search_t = pr::search_engine<Float, dst_t, order>;

        callback.set_euclidean_distance(true);

        const dst_t dist{ queue };
        const search_t search{ queue, train_data, train_block, dist };
        search(query_data, callback, infer_block, neighbor_count).wait_and_throw();
    }
    else if (is_minkowski_distance) {
        using met_t = pr::lp_metric<Float>;
        using dst_t = pr::lp_distance<Float>;
        [[maybe_unused]] constexpr auto order = get_ndorder(train_data);
        using search_t = pr::search_engine<Float, dst_t, order>;

        const dst_t dist{ queue, met_t(distance_impl->get_degree()) };
        const search_t search{ queue, train_data, train_block, dist };
        search(query_data, callback, infer_block, neighbor_count).wait_and_throw();
    }

    auto result = infer_result<Task>{}.set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        if constexpr (detail::is_not_search_v<Task>) {
            result = result.set_responses(homogen_table::wrap(arr_responses, infer_row_count, 1));
        }
    }

    if (desc.get_result_options().test(result_options::indices)) {
        result =
            result.set_indices(homogen_table::wrap(arr_indices, infer_row_count, neighbor_count));
    }

    if (desc.get_result_options().test(result_options::distances)) {
        result = result.set_distances(
            homogen_table::wrap(arr_distances, infer_row_count, neighbor_count));
    }

    return result;
}

template <typename Float, typename Task>
static infer_result<Task> call_kernel(const context_gpu& ctx,
                                      const descriptor_t<Task>& desc,
                                      const table& infer,
                                      const model<Task>& m) {
    const auto trained_model = dynamic_cast_to_knn_model<Task, brute_force_model_impl<Task>>(m);
    const auto train = trained_model->get_data();
    const bool cm_train = is_col_major(train);
    const bool cm_query = is_col_major(infer);
    if (cm_train) {
        if (cm_query)
            return kernel<Float, Task, true, true>(ctx, desc, infer, m);
        else
            return kernel<Float, Task, true, false>(ctx, desc, infer, m);
    }
    else {
        if (cm_query)
            return kernel<Float, Task, false, true>(ctx, desc, infer, m);
        else
            return kernel<Float, Task, false, false>(ctx, desc, infer, m);
    }
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_gpu& ctx,
                                const descriptor_t<Task>& desc,
                                const infer_input<Task>& input) {
    return call_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float, typename Task>
struct infer_kernel_gpu<Float, method::brute_force, Task> {
    infer_result<Task> operator()(const context_gpu& ctx,
                                  const descriptor_t<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, method::brute_force, task::classification>;
template struct infer_kernel_gpu<double, method::brute_force, task::classification>;
template struct infer_kernel_gpu<float, method::brute_force, task::regression>;
template struct infer_kernel_gpu<double, method::brute_force, task::regression>;
template struct infer_kernel_gpu<float, method::brute_force, task::search>;
template struct infer_kernel_gpu<double, method::brute_force, task::search>;

} // namespace oneapi::dal::knn::backend
