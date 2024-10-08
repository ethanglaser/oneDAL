/* file: x86_64_kernel_defines.h */
/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#ifndef __x86_64_KERNEL_DEFINES_H__
#define __x86_64_KERNEL_DEFINES_H__

#if defined(DAAL_KERNEL_SSE2)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID           daal::sse2
    #define DAAL_KERNEL_SSE2_ONLY(something)                   , something
    #define DAAL_KERNEL_SSE2_ONLY_CODE(...)                    __VA_ARGS__
    #define DAAL_KERNEL_SSE2_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sse2, __VA_ARGS__)
    #define DAAL_KERNEL_SSE2_CONTAINER1(ContainerTemplate, ...) \
        extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sse2, __VA_ARGS__);
    #define DAAL_KERNEL_SSE2_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, sse2, __VA_ARGS__)
#else
    #define DAAL_KERNEL_SSE2_ONLY(something)
    #define DAAL_KERNEL_SSE2_ONLY_CODE(...)
    #define DAAL_KERNEL_SSE2_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_SSE2_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_SSE2_CONTAINER_CASE(ContainerTemplate, ...)
#endif

#if defined(DAAL_KERNEL_SSE42)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID            daal::sse42
    #define DAAL_KERNEL_SSE42_ONLY(something)                   , something
    #define DAAL_KERNEL_SSE42_ONLY_CODE(...)                    __VA_ARGS__
    #define DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sse42, __VA_ARGS__)
    #define DAAL_KERNEL_SSE42_CONTAINER1(ContainerTemplate, ...) \
        extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, sse42, __VA_ARGS__);
    #define DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, sse42, __VA_ARGS__)
#else
    #define DAAL_KERNEL_SSE42_ONLY(something)
    #define DAAL_KERNEL_SSE42_ONLY_CODE(...)
    #define DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_SSE42_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_SSE42_CONTAINER_CASE(ContainerTemplate, ...)
#endif

#if defined(DAAL_KERNEL_AVX2)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID           daal::avx2
    #define DAAL_KERNEL_AVX2_ONLY(something)                   , something
    #define DAAL_KERNEL_AVX2_ONLY_CODE(...)                    __VA_ARGS__
    #define DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx2, __VA_ARGS__)
    #define DAAL_KERNEL_AVX2_CONTAINER1(ContainerTemplate, ...) \
        extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx2, __VA_ARGS__);
    #define DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, avx2, __VA_ARGS__)
#else
    #define DAAL_KERNEL_AVX2_ONLY(something)
    #define DAAL_KERNEL_AVX2_ONLY_CODE(...)
    #define DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX2_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX2_CONTAINER_CASE(ContainerTemplate, ...)
#endif

#if defined(DAAL_KERNEL_AVX512)
    #undef DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID
    #define DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID             daal::avx512
    #define DAAL_KERNEL_AVX512_ONLY(something)                   , something
    #define DAAL_KERNEL_AVX512_ONLY_CODE(...)                    __VA_ARGS__
    #define DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, ...) , DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx512, __VA_ARGS__)
    #define DAAL_KERNEL_AVX512_CONTAINER1(ContainerTemplate, ...) \
        extern template class DAAL_KERNEL_CONTAINER_TEMPL(ContainerTemplate, avx512, __VA_ARGS__);
    #define DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, ...) DAAL_KERNEL_CONTAINER_CASE(ContainerTemplate, avx512, __VA_ARGS__)
#else
    #define DAAL_KERNEL_AVX512_ONLY(something)
    #define DAAL_KERNEL_AVX512_ONLY_CODE(...)
    #define DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX512_CONTAINER1(ContainerTemplate, ...)
    #define DAAL_KERNEL_AVX512_CONTAINER_CASE(ContainerTemplate, ...)
#endif

#endif
