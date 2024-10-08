/* file: gbt_classification_predict_result_fpt.cpp */
/*******************************************************************************
* Copyright 2024 Intel Corporation
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
//  Implementation of the gradient boosted trees classification algorithm interface
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_classification_predict_types.h"
#include "data_management/data/homogen_numeric_table.h"
#include "src/services/daal_strings.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace prediction
{
using namespace daal::services;

template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par, const int method)
{
    using algorithms::classifier::computeClassLabels;
    using algorithms::classifier::computeClassProbabilities;
    using algorithms::classifier::computeClassLogProbabilities;
    using algorithms::classifier::prediction::data;

    const Input * algInput                   = (static_cast<const Input *>(input));
    data_management::NumericTablePtr dataPtr = algInput->get(data);
    DAAL_CHECK_EX(dataPtr.get(), ErrorNullInputNumericTable, ArgumentName, dataStr());
    services::Status s;
    const size_t nVectors = dataPtr->getNumberOfRows();

    const Parameter * classificationParameter = static_cast<const Parameter *>(par);

    if (classificationParameter->resultsToEvaluate & computeClassLabels)
    {
        size_t nColumnsToAllocate = 1;
        if (classificationParameter->resultsToCompute & shapContributions)
        {
            const size_t nColumns = dataPtr->getNumberOfColumns();
            nColumnsToAllocate    = nColumns + 1;
        }
        else if (classificationParameter->resultsToCompute & shapInteractions)
        {
            const size_t nColumns = dataPtr->getNumberOfColumns();
            nColumnsToAllocate    = (nColumns + 1) * (nColumns + 1);
        }

        Argument::set(prediction, data_management::HomogenNumericTable<algorithmFPType>::create(nColumnsToAllocate, nVectors,
                                                                                                data_management::NumericTableIface::doAllocate, &s));
    }

    if (classificationParameter->resultsToEvaluate & computeClassProbabilities)
    {
        Argument::set(probabilities, data_management::HomogenNumericTable<algorithmFPType>::create(
                                         classificationParameter->nClasses, nVectors, data_management::NumericTableIface::doAllocate, &s));
    }

    if (classificationParameter->resultsToEvaluate & computeClassLogProbabilities)
    {
        Argument::set(logProbabilities, data_management::HomogenNumericTable<algorithmFPType>::create(
                                            classificationParameter->nClasses, nVectors, data_management::NumericTableIface::doAllocate, &s));
    }

    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const daal::algorithms::Parameter * par,
                                                                    const int method);

} // namespace prediction
} // namespace classification
} // namespace gbt
} // namespace algorithms
} // namespace daal
