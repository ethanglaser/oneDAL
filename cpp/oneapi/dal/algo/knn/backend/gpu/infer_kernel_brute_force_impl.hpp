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

#pragma once

//TODO: identify unnecessary includes

// from cpp
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/regression.hpp"
#include "oneapi/dal/backend/primitives/search.hpp"
#include "oneapi/dal/backend/primitives/selection.hpp"
#include "oneapi/dal/backend/primitives/voting.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

//#include "oneapi/dal/table/row_accessor.hpp"

//#include "oneapi/dal/detail/common.hpp"
// from examples
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/communicator.hpp"


#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::knn::backend {

namespace de = ::oneapi::dal::detail;
namespace bk = ::oneapi::dal::backend;
namespace pr = ::oneapi::dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;

using idx_t = std::int32_t;

using dal::backend::context_gpu;

template <typename Task>
using descriptor_t = detail::descriptor_base<Task>;

using voting_t = ::oneapi::dal::knn::voting_mode;

using daal_distance_t = daal::algorithms::internal::PairwiseDistanceType;

template <typename Task>
struct task_to_response_map {
    using type = int;
};

template <>
struct task_to_response_map<task::regression> {
    using type = float;
};

template <>
struct task_to_response_map<task::classification> {
    using type = std::int32_t;
};

template <typename Task>
using response_t = typename task_to_response_map<Task>::type;

template <typename Float, typename Task>
class knn_callback {
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using Task = std::tuple_element_t<2, TestType>;
    using dst_t = Float;
    using idx_t = std::int32_t;
    using res_t = response_t<Task>;
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;

    using uniform_voting_t = std::unique_ptr<pr::uniform_voting<res_t>>;
    using distance_voting_t = std::unique_ptr<pr::distance_voting<dst_t>>;
    using uniform_regression_t = std::unique_ptr<pr::uniform_regression<res_t>>;
    using distance_regression_t = std::unique_ptr<pr::distance_regression<dst_t>>;

public:
    knn_callback(sycl::queue& q,
                 comm_t c,
                 result_option_id results,
                 std::int64_t query_block,
                 std::int64_t query_length,
                 std::int64_t k_neighbors) {}

    auto& set_euclidean_distance(bool is_euclidean_distance);
    auto& set_inp_responses(const pr::ndview<res_t, 1>& inp_responses);
    auto& set_uniform_voting(uniform_voting_t voting);
    auto& set_distance_voting(distance_voting_t voting);
    auto& set_uniform_regression(uniform_regression_t regression);
    auto& set_distance_regression(distance_regression_t regression);
    auto& set_responses(const array<res_t>& responses);
    auto& set_indices(const array<idx_t>& indices);
    auto& set_distances(array<Float>& distances);
    auto get_blocking() const;
    sycl::event operator()(std::int64_t qb_id,
                           pr::ndview<idx_t, 2>& inp_indices,
                           pr::ndview<Float, 2>& inp_distances,
                           const bk::event_vector& deps = {});

private:
    sycl::queue& queue_;
    comm_t comm_;
    const result_option_id result_options_;
    const std::int64_t query_block_, query_length_, k_neighbors_;
    pr::ndview<res_t, 1> inp_responses_;
    pr::ndarray<res_t, 2> temp_resp_;
    pr::ndarray<res_t, 1> responses_;
    pr::ndarray<Float, 2> distances_;
    pr::ndarray<idx_t, 2> indices_;
    uniform_voting_t uniform_voting_;
    distance_voting_t distance_voting_;
    uniform_regression_t uniform_regression_;
    distance_regression_t distance_regression_;
    bool compute_sqrt_ = false;
};

} // namespace oneapi::dal::knn::backend

#endif // ONEDAL_DATA_PARALLEL
