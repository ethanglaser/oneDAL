/* file: predict_batch.cpp */
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

#include <jni.h>

#include "daal.h"
#include "com/intel/daal/common_helpers.h"
#include "com_intel_daal_algorithms_logitboost_prediction_PredictionBatch.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::logitboost::prediction;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logitboost_prediction_PredictionBatch_cInit(JNIEnv * env, jobject thisObj, jint prec,
                                                                                                   jint method, jlong nClasses)
{
    return jniBatch<logitboost::prediction::Method, Batch, defaultDense>::newObj(prec, method, (size_t)nClasses);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logitboost_prediction_PredictionBatch_cInitParameter(JNIEnv * env, jobject thisObj,
                                                                                                            jlong algAddr, jint prec, jint method)
{
    return jniBatch<logitboost::prediction::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logitboost_prediction_PredictionBatch_cClone(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                    jint prec, jint method)
{
    return jniBatch<logitboost::prediction::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}