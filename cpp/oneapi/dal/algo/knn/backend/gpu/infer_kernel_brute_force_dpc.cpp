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

#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel_impl.hpp"


namespace oneapi::dal::knn::backend {

using dal::backend::context_gpu;

template<typename Task>
using descriptor_t = detail::descriptor_base<Task>;
template<typename Task>
using model_t = model<Task>;

template <typename Float, typename Task>
static infer_result<Task> kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const table& infer,
                                 const model_t& m) {
    return infer_kernel_knn_bf_impl<Float, Task>(ctx)(desc, infer, m);
                                 }

} // namespace oneapi::dal::knn::backend
