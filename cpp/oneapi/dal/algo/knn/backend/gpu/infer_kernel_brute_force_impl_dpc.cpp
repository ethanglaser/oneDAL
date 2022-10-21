/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel_brute_force_impl.hpp"

namespace oneapi::dal::knn::backend {

using dst_t = Float;
using idx_t = std::int32_t;
using res_t = response_t<Task>;
using comm_t = bk::communicator<spmd::device_memory_access::usm>;

using uniform_voting_t = std::unique_ptr<pr::uniform_voting<res_t>>;
using distance_voting_t = std::unique_ptr<pr::distance_voting<dst_t>>;
using uniform_regression_t = std::unique_ptr<pr::uniform_regression<res_t>>;
using distance_regression_t = std::unique_ptr<pr::distance_regression<dst_t>>;

knn_callback<Float, Task>::knn_callback(sycl::queue& q,
                comm_t c,
                result_option_id results,
                std::int64_t query_block,
                std::int64_t query_length,
                std::int64_t k_neighbors)
        : queue_(q),
            comm_(c),
            result_options_(results),
            query_block_(query_block),
            query_length_(query_length),
            k_neighbors_(k_neighbors) {
    if (result_options_.test(result_options::responses)) {
        this->temp_resp_ = pr::ndarray<res_t, 2>::empty(q,
                                                        { query_block, k_neighbors },
                                                        sycl::usm::alloc::device);
    }
}

template <typename Float, typename Task>
auto& knn_callback<Float, Task>::set_euclidean_distance(bool is_euclidean_distance) {
    this->compute_sqrt_ = is_euclidean_distance;
    return *this;
}

template <typename Float, typename Task>
auto& knn_callback<Float, Task>::set_inp_responses(const pr::ndview<res_t, 1>& inp_responses) {
    if (result_options_.test(result_options::responses)) {
        this->inp_responses_ = inp_responses;
    }
    return *this;
}

template <typename T = Task, typename = detail::enable_if_classification_t<T>>
auto& knn_callback<Float, Task>::set_uniform_voting(uniform_voting_t voting) {
    this->uniform_voting_ = std::move(voting);
    return *this;
}

template <typename T = Task, typename = detail::enable_if_classification_t<T>>
auto& knn_callback<Float, Task>::set_distance_voting(distance_voting_t voting) {
    this->distance_voting_ = std::move(voting);
    return *this;
}

template <typename T = Task, typename = detail::enable_if_regression_t<T>>
auto& knn_callback<Float, Task>::set_uniform_regression(uniform_regression_t regression) {
    this->uniform_regression_ = std::move(regression);
    return *this;
}

template <typename T = Task, typename = detail::enable_if_regression_t<T>>
auto& knn_callback<Float, Task>::set_distance_regression(distance_regression_t regression) {
    this->distance_regression_ = std::move(regression);
    return *this;
}

template <typename Float, typename Task>
auto& knn_callback<Float, Task>::set_responses(const array<res_t>& responses) {
    if (result_options_.test(result_options::responses)) {
        ONEDAL_ASSERT(responses.get_count() == query_length_);
        this->responses_ = pr::ndarray<res_t, 1>::wrap_mutable(responses, query_length_);
    }
    return *this;
}

template <typename Float, typename Task>
auto& knn_callback<Float, Task>::set_indices(const array<idx_t>& indices) {
    if (result_options_.test(result_options::indices)) {
        ONEDAL_ASSERT(indices.get_count() ==
                        de::check_mul_overflow(query_length_, k_neighbors_));
        this->indices_ =
            pr::ndarray<idx_t, 2>::wrap_mutable(indices, { query_length_, k_neighbors_ });
    }
    return *this;
}

template <typename Float, typename Task>
auto& knn_callback<Float, Task>::set_distances(array<Float>& distances) {
    if (result_options_.test(result_options::distances)) {
        ONEDAL_ASSERT(distances.get_count() ==
                        de::check_mul_overflow(query_length_, k_neighbors_));
        this->distances_ =
            pr::ndarray<Float, 2>::wrap_mutable(distances, { query_length_, k_neighbors_ });
    }
    return *this;
}

template <typename Float, typename Task>
auto knn_callback<Float, Task>::get_blocking() const {
    return bk::uniform_blocking(query_length_, query_block_);
}

//TODO: does every function need a template header
// Note: `inp_distances` can be modified if
// metric is Euclidean
template <typename Float, typename Task>
sycl::event knn_callback<Float, Task>::operator()(std::int64_t qb_id,
                        pr::ndview<idx_t, 2>& inp_indices,
                        pr::ndview<Float, 2>& inp_distances,
                        const bk::event_vector& deps = {}) {
    sycl::event copy_indices, copy_distances, comp_responses;
    const auto blocking = this->get_blocking();

    const auto from = blocking.get_block_start_index(qb_id);
    const auto to = blocking.get_block_end_index(qb_id);

    if (result_options_.test(result_options::indices)) {
        auto out_block = indices_.get_row_slice(from, to);
        copy_indices = copy(queue_, out_block, inp_indices, deps);
    }

    if (result_options_.test(result_options::distances)) {
        auto out_block = distances_.get_row_slice(from, to);
        if (this->compute_sqrt_) {
            copy_distances = copy_with_sqrt(queue_, inp_distances, out_block, deps);
        }
        else {
            copy_distances = copy(queue_, out_block, inp_distances, deps);
        }
    }

    if (result_options_.test(result_options::responses)) {
        using namespace bk;
        auto out_block = responses_.get_slice(from, to);
        const auto ndeps = deps + copy_indices + copy_distances;
        auto temp_resp = temp_resp_.get_row_slice(0, to - from);
        auto s_event = select_indexed(queue_, inp_indices, inp_responses_, temp_resp, ndeps);

        // One and only one functor can be initialized
        ONEDAL_ASSERT((bool(distance_voting_) + bool(uniform_voting_) +
                        bool(distance_regression_) + bool(uniform_regression_)) == 1);

        if constexpr (std::is_same_v<Task, task::classification>) {
            if (uniform_voting_) {
                comp_responses = uniform_voting_->operator()(temp_resp, out_block, { s_event });
            }

            if (distance_voting_) {
                sycl::event sqrt_event;

                if (this->compute_sqrt_) {
                    sqrt_event = copy_with_sqrt(queue_, inp_distances, inp_distances, deps);
                }

                comp_responses = distance_voting_->operator()(temp_resp,
                                                                inp_distances,
                                                                out_block,
                                                                { sqrt_event, s_event });
            }
        }

        if constexpr (std::is_same_v<Task, task::regression>) {
            if (uniform_regression_) {
                comp_responses =
                    uniform_regression_->operator()(temp_resp, out_block, { s_event });
            }

            if (distance_regression_) {
                sycl::event sqrt_event;

                if (this->compute_sqrt_) {
                    sqrt_event = copy_with_sqrt(queue_, inp_distances, inp_distances, deps);
                }

                comp_responses = distance_regression_->operator()(temp_resp,
                                                                    inp_distances,
                                                                    out_block,
                                                                    { sqrt_event, s_event });
            }
        }
    }

    sycl::event::wait_and_throw({ copy_indices, copy_distances, comp_responses });
    return sycl::event();
}

} // namespace oneapi::dal::knn::backend

#endif // ONEDAL_DATA_PARALLEL
