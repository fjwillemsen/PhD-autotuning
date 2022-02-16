#!/usr/bin/env python
import sys
import kernel_tuner as kt
import numpy as np
import argparse
import json
import os
from collections import OrderedDict

from .common import reg_observer

metrics = OrderedDict()
metrics["registers"] = lambda p: p["num_regs"]

# CUDA source code
dir_name = os.path.dirname(os.path.realpath(__file__)) + '/'
kernels_src = dir_name + 'rte_solver_kernels.cu'

ref_kernels_src = dir_name + 'reference_kernels/rte_solver_kernels.cu'

include = dir_name + '/include'
cp = ['-I{}'.format(include)]

# Settings
type_int = np.int32
type_float = np.float64
type_bool = np.int8    # = default without `RTE_RRTMGP_USE_CBOOL`

str_float = 'float' if type_float is np.float32 else 'double'

ncol = type_int(512)
nlay = type_int(140)
ngpt = type_int(224)

top_at_1 = type_bool(0)

opt_size = ncol * nlay * ngpt
flx_size = ncol * (nlay + 1) * ngpt
alb_size = ncol * ngpt

# Input arrays; for this kernel, the values don't matter..
flux_up = np.zeros(flx_size, dtype=type_float)
flux_dn = np.zeros(flx_size, dtype=type_float)
flux_dir = np.zeros(flx_size, dtype=type_float)

sfc_alb_dir = np.zeros(alb_size, dtype=type_float)
sfc_alb_dif = np.zeros(alb_size, dtype=type_float)

# Output arrays
r_dif = np.zeros(opt_size, dtype=type_float)
t_dif = np.zeros(opt_size, dtype=type_float)
r_dir = np.zeros(opt_size, dtype=type_float)
t_dir = np.zeros(opt_size, dtype=type_float)
t_noscat = np.zeros(opt_size, dtype=type_float)
source_up = np.zeros(opt_size, dtype=type_float)
source_dn = np.zeros(opt_size, dtype=type_float)
source_sfc = np.zeros(alb_size, dtype=type_float)
albedo = np.zeros(flx_size, dtype=type_float)
src = np.zeros(flx_size, dtype=type_float)
denom = np.zeros(opt_size, dtype=type_float)

source_kernel_args = [ncol, nlay, ngpt, top_at_1, r_dir, t_dir, t_noscat, sfc_alb_dir, source_up, source_dn, source_sfc, flux_dir]

adding_ref_kernel_args = [
    ncol, nlay, ngpt, top_at_1, sfc_alb_dif, r_dif, t_dif, source_dn, source_up, source_sfc, flux_up, flux_dn, flux_dir, albedo, src, denom
]

adding_kernel_args = [ncol, nlay, ngpt, top_at_1, sfc_alb_dif, r_dif, t_dif, source_dn, source_up, source_sfc, flux_up, flux_dn, flux_dir, albedo, src, denom]

problem_size = (ncol, ngpt)
source_kernel_name = f'sw_source_kernel<{str_float}, {top_at_1}>'
source_ref_kernel_name = f'sw_source_kernel<{str_float}>'
adding_kernel_name = f'sw_adding_kernel<{str_float}, {top_at_1}>'
adding_ref_kernel_name = f'sw_adding_kernel<{str_float}>'


def tune(device_name, strategy="bayes_opt", strategy_options=None, verbose=True, quiet=False, simulation_mode=True):

    tune_params = OrderedDict()
    tune_params['RTE_RRTMGP_USE_CBOOL'] = [1]
    tune_params['block_size_x'] = [16 * i for i in range(1, 33)]
    tune_params['block_size_y'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    #additional parameters for adding kernel
    #tune_params["block_size_x"] = [32]
    #tune_params["block_size_y"] = [7]
    tune_params['loop_unroll_factor_nlay'] = [0] + [i for i in range(1, nlay + 1) if nlay // i == nlay / i]
    tune_params['recompute_denom'] = [0, 1]
    #unrolling the second loop made no difference
    #tune_params['loop_unroll_factor_p1nlay'] = [0, 1, 3]

    # adding_result = kt.run_kernel(adding_ref_kernel_name, ref_kernels_src, problem_size, adding_ref_kernel_args,
    #                               dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=512, block_size_y=1), compiler_options=cp)

    adding_answer = [None for _ in adding_kernel_args]

    # adding_answer[-6] = adding_result[-6]    # flux_up
    # adding_answer[-5] = adding_result[-5]    # flux_dn
    # adding_answer[-4] = adding_result[-4]    # flux_dir
    # adding_answer[-3] = adding_result[-3]    # albedo
    # adding_answer[-2] = adding_result[-2]    # src
    # #adding_answer[-1] = adding_result[-1] # denom #removed

    result, env = kt.tune_kernel(adding_kernel_name, kernels_src, problem_size, adding_kernel_args, tune_params, compiler_options=cp, answer=adding_answer,
                                 atol=1e-14, observers=[reg_observer], metrics=metrics, iterations=32, cache="cache_files/adding_kernel_" + device_name,
                                 strategy=strategy, strategy_options=strategy_options, simulation_mode=simulation_mode, verbose=verbose, quiet=quiet)

    return result, env


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./adding_kernel_wrapper.py [device name]")
        exit(1)

    device_name = sys.argv[1]

    tune(device_name)
