#===============================================================================
# Copyright contributors to the oneDAL project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

#++
#  g++ definitions for makefile
#--

include dev/make/compiler_definitions/gnu.mk

PLATs.gnu = lnxarm

OPTFLAGS_SUPPORTED := O0 O1 O2 O3 Os Ofast Og Oz

ifneq (,$(filter $(OPTFLAG),$(OPTFLAGS_SUPPORTED)))
else
    $(error Invalid OPTFLAG '$(OPTFLAG)' for $(COMPILER). Supported: $(OPTFLAGS_SUPPORTED))
endif

ifeq ($(filter $(OPTFLAG),O0 Og),$(OPTFLAG))
    -optlevel.gnu = -$(OPTFLAG)
else
    -optlevel.gnu = -$(OPTFLAG) -D_FORTIFY_SOURCE=2
endif

COMPILER.all.gnu =  ${CXX} -march=armv8-a+sve -fopenmp-simd -ftree-vectorize -fwrapv -fno-strict-overflow -fno-delete-null-pointer-checks \
                    -DDAAL_REF -DONEDAL_REF -DDAAL_CPU=sve -Werror -Wreturn-type $(if $(RNG_OPENRNG), -DOPENRNG_BACKEND)

link.dynamic.all.gnu = ${CXX} -march=native

COMPILER.lnx.gnu = $(COMPILER.all.gnu)
link.dynamic.lnx.gnu = $(link.dynamic.all.gnu)
pedantic.opts.lnx.gnu = $(pedantic.opts.all.gnu)

a8sve_OPT.gnu = $(-Q)march=armv8-a+sve
