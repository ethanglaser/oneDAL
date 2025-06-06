/* file: kmeans_lloyd_batch_impl.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/*
//++
//  Implementation of Lloyd method for K-means algorithm.
//--
*/

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "src/threading/threading.h"
#include "services/daal_defines.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_defines.h"

#include "src/algorithms/kmeans/kmeans_lloyd_impl.i"
#include "src/algorithms/kmeans/kmeans_lloyd_postprocessing.h"

#include "src/services/service_profiler.h"

using namespace daal::internal;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace internal
{
template <Method method, typename algorithmFPType, CpuType cpu>
Status KMeansBatchKernel<method, algorithmFPType, cpu>::compute(const NumericTable * const * a, const NumericTable * const * r, const Parameter * par)
{
    Status s;
    NumericTable * ntData  = const_cast<NumericTable *>(a[0]);
    const size_t nIter     = par->maxIterations;
    const size_t n         = ntData->getNumberOfRows();
    const size_t p         = ntData->getNumberOfColumns();
    const size_t nClusters = par->nClusters;
    int result             = 0;

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(int));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, p);
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters * p, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, p, sizeof(algorithmFPType));

    TArray<int, cpu> clusterS0(nClusters);
    TArray<algorithmFPType, cpu> clusterS1(nClusters * p);
    DAAL_CHECK(clusterS0.get() && clusterS1.get(), services::ErrorMemoryAllocationFailed);

    ReadRows<algorithmFPType, cpu> mtInClusters(*const_cast<NumericTable *>(a[1]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtInClusters);
    algorithmFPType * inClusters = const_cast<algorithmFPType *>(mtInClusters.get());

    WriteOnlyRows<algorithmFPType, cpu> mtClusters(const_cast<NumericTable *>(r[0]), 0, nClusters);
    DAAL_CHECK_BLOCK_STATUS(mtClusters);
    algorithmFPType * clusters = mtClusters.get();

    TArray<algorithmFPType, cpu> tClusters;
    if (clusters == nullptr && nIter != 0)
    {
        tClusters.reset(nClusters * p);
        clusters = tClusters.get();
    }

    NumericTable * assignmetsNT = nullptr;
    NumericTablePtr assignmentsPtr;
    if (r[1])
    {
        assignmetsNT = const_cast<NumericTable *>(r[1]);
    }
    else if (par->resultsToEvaluate & computeExactObjectiveFunction)
    {
        assignmentsPtr = HomogenNumericTableCPU<int, cpu>::create(1, n, &s);
        DAAL_CHECK_MALLOC(s);
        assignmetsNT = assignmentsPtr.get();
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, p, sizeof(double));

    TArray<double, cpu> dS1(method == defaultDense ? p : 0);
    if (method == defaultDense)
    {
        DAAL_CHECK(dS1.get(), services::ErrorMemoryAllocationFailed);
    }

    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(algorithmFPType));
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nClusters, sizeof(size_t));

    TArray<algorithmFPType, cpu> cValues(nClusters);
    TArray<size_t, cpu> cIndices(nClusters);

    algorithmFPType oldTargetFunc(0.0);

    size_t blockSize = 0;
    DAAL_SAFE_CPU_CALL((blockSize = BSHelper<method, algorithmFPType, cpu>::kmeansGetBlockSize(n, p, nClusters)), (blockSize = 512))

    size_t kIter;

    for (kIter = 0; kIter < nIter; kIter++)
    {
        auto task = TaskKMeansLloyd<algorithmFPType, cpu>::create(p, nClusters, inClusters, blockSize);
        DAAL_CHECK(task.get(), services::ErrorMemoryAllocationFailed);
        {
            DAAL_PROFILER_TASK(addNTToTaskThreaded);
            /* For the last iteration we do not need to recount of assignmets */
            s = task->template addNTToTaskThreaded<method>(ntData, nullptr, blockSize, assignmetsNT && (kIter == nIter - 1) ? assignmetsNT : nullptr);
        }

        if (!s)
        {
            task->kmeansClearClusters(&oldTargetFunc);
            break;
        }

        {
            DAAL_PROFILER_TASK(kmeansPartialReduceCentroids);
            task->template kmeansComputeCentroids<method>(clusterS0.get(), clusterS1.get(), dS1.get());
        }

        size_t cNum;
        DAAL_CHECK_STATUS(s, task->kmeansComputeCentroidsCandidates(cValues.get(), cIndices.get(), cNum));
        size_t cPos = 0;

        algorithmFPType newCentersGoalFunc = (algorithmFPType)0.0;
        algorithmFPType l2Norm             = (algorithmFPType)0.0;
        {
            DAAL_PROFILER_TASK(kmeansMergeReduceCentroids);

            for (size_t i = 0; i < nClusters; i++)
            {
                if (clusterS0[i] > 0)
                {
                    const algorithmFPType coeff = 1.0 / clusterS0[i];

                    PRAGMA_FORCE_SIMD
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t j = 0; j < p; j++)
                    {
                        const algorithmFPType newCluster = clusterS1[i * p + j] * coeff;
                        const algorithmFPType dist       = clusters[i * p + j] - newCluster;
                        l2Norm += dist * dist;
                        clusters[i * p + j] = newCluster;
                    }
                }
                else
                {
                    DAAL_CHECK(cPos < cNum, services::ErrorKMeansNumberOfClustersIsTooLarge);
                    newCentersGoalFunc += cValues[cPos];
                    ReadRows<algorithmFPType, cpu> mtRow(ntData, cIndices[cPos], 1);
                    const algorithmFPType * row = mtRow.get();

                    PRAGMA_FORCE_SIMD
                    PRAGMA_VECTOR_ALWAYS
                    for (size_t j = 0; j < p; j++)
                    {
                        const algorithmFPType dist = clusters[i * p + j] - row[j];
                        l2Norm += dist * dist;
                    }
                    result |=
                        daal::services::internal::daal_memcpy_s(&clusters[i * p], p * sizeof(algorithmFPType), row, p * sizeof(algorithmFPType));
                    cPos++;
                }
            }
        }
        {
            DAAL_PROFILER_TASK(kmeansUpdateObjectiveFunction);
            if (par->accuracyThreshold > (algorithmFPType)0.0)
            {
                algorithmFPType newTargetFunc = (algorithmFPType)0.0;

                task->kmeansClearClusters(&newTargetFunc);
                newTargetFunc -= newCentersGoalFunc;

                if (l2Norm < par->accuracyThreshold)
                {
                    kIter++;
                    break;
                }

                oldTargetFunc = newTargetFunc;
            }
            else
            {
                task->kmeansClearClusters(&oldTargetFunc);
                oldTargetFunc -= newCentersGoalFunc;
            }
        }
        inClusters = clusters;
    }

    if (!nIter)
    {
        clusters = inClusters;
    }

    if (par->resultsToEvaluate & computeAssignments || par->assignFlag || par->resultsToEvaluate & computeExactObjectiveFunction)
    {
        PostProcessing<method, algorithmFPType, cpu>::computeAssignments(p, nClusters, clusters, ntData, nullptr, assignmetsNT, blockSize);
    }

    if (par->resultsToEvaluate & computeExactObjectiveFunction)
    {
        WriteOnlyRows<algorithmFPType, cpu> mtTarget(*const_cast<NumericTable *>(r[2]), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(mtTarget);
        algorithmFPType exactTargetFunc = algorithmFPType(0);
        PostProcessing<method, algorithmFPType, cpu>::computeExactObjectiveFunction(p, nClusters, clusters, ntData, nullptr, assignmetsNT,
                                                                                    exactTargetFunc, blockSize);

        *mtTarget.get() = exactTargetFunc;
    }
    if (r[3])
    {
        WriteOnlyRows<int, cpu> mtIterations(*const_cast<NumericTable *>(r[3]), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(mtIterations);
        *mtIterations.get() = kIter;
    }
    return (!result) ? s : services::Status(services::ErrorMemoryCopyFailedInternal);
}

} // namespace internal
} // namespace kmeans
} // namespace algorithms
} // namespace daal
