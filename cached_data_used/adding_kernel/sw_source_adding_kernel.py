#!/usr/bin/env python
import kernel_tuner as kt
import numpy as np
import argparse
import json
import os
from collections import OrderedDict

from common import reg_observer

metrics = OrderedDict()
metrics["registers"] = lambda p: p["num_regs"]

# CUDA source code
dir_name = os.path.dirname(os.path.realpath(__file__)) + '/'
kernels_src = dir_name + 'rte_solver_kernels.cu'

ref_kernels_src = dir_name + 'reference_kernels/rte_solver_kernels.cu'

include = dir_name + '/include'
cp = ['-I{}'.format(include)]


# Parse command line arguments
def parse_command_line():
    parser = argparse.ArgumentParser(description='sw_source_adding_kernel()')
    parser.add_argument('--tune', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--best_configuration', default=False, action='store_true')
    parser.add_argument('--block_size_x', type=int, default=32)
    parser.add_argument('--block_size_y', type=int, default=4)
    return parser.parse_args()


# Run one instance of the kernel and test output
def run_and_test(params: dict):
    #print('Running {} [block_size_x: {}, block_size_y: {}]'.format(
    #        kernel_name, params['block_size_x'], params['block_size_y']))

    print("Testing source kernel")

    source_ref_result = kt.run_kernel(source_ref_kernel_name, ref_kernels_src, problem_size, source_kernel_args,
                                      dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=512, block_size_y=1), compiler_options=cp)

    source_result = kt.run_kernel(source_kernel_name, kernels_src, problem_size, source_kernel_args,
                                  dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=64, block_size_y=11), compiler_options=cp)

    outputs = ["flux_dir", "source_sfc", "source_dn", "source_up"]
    for i, o in enumerate(outputs):
        okay = np.allclose(source_result[-1 - i], source_ref_result[-1 - i], atol=1e-14)
        max_diff = np.abs(source_result[-1 - i], source_ref_result[-1 - i]).max()
        if okay:
            print(f'results for {o}: OKAY!')
        else:
            print(f'results for {o}: NOT OKAY, {max_diff=}')

    print("Testing adding kernel")

    adding_ref_result = kt.run_kernel(adding_ref_kernel_name, ref_kernels_src, problem_size, adding_ref_kernel_args,
                                      dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=512, block_size_y=1), compiler_options=cp)

    adding_result = kt.run_kernel(adding_kernel_name, kernels_src, problem_size, adding_kernel_args,
                                  dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=32, block_size_y=4, loop_unroll_factor_nlay=140), compiler_options=cp)

    outputs = ["flux_dir", "flux_dn", "flux_up"]
    for i, o in enumerate(outputs):
        okay = np.allclose(adding_result[-4 - i], adding_ref_result[-4 - i], atol=1e-14)
        max_diff = np.abs(adding_result[-4 - i], adding_ref_result[-4 - i]).max()
        if okay:
            print(f'results for {o}: OKAY!')
        else:
            print(f'results for {o}: NOT OKAY, {max_diff=}')


# Tuning the kernel
def tune():

    source_result = kt.run_kernel(source_ref_kernel_name, ref_kernels_src, problem_size, source_kernel_args,
                                  dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=512, block_size_y=1), compiler_options=cp)

    source_answer = [None for _ in source_kernel_args]
    source_answer[-4] = source_result[-4]    #source_up
    source_answer[-3] = source_result[-3]    #source_dn
    source_answer[-2] = source_result[-2]    #source_sfc
    source_answer[-1] = source_result[-1]    #flux_dir

    tune_params = OrderedDict()
    tune_params['RTE_RRTMGP_USE_CBOOL'] = [1]
    tune_params['block_size_x'] = [16 * i for i in range(1, 33)]
    tune_params['block_size_y'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    print(f"Tuning {source_kernel_name}")
    #result, env = kt.tune_kernel(
    #        source_kernel_name, kernels_src, problem_size,
    #        source_kernel_args, tune_params, compiler_options=cp,
    #        answer=source_answer, atol=1e-14,
    #        verbose=True, observers=[reg_observer], metrics=metrics, iterations=32)

    #with open('timings_sw_source_kernel.json', 'w') as fp:
    #    json.dump(result, fp)

    #additional parameters for adding kernel
    #tune_params["block_size_x"] = [32]
    #tune_params["block_size_y"] = [7]
    tune_params['loop_unroll_factor_nlay'] = [0] + [i for i in range(1, nlay + 1) if nlay // i == nlay / i]
    tune_params['recompute_denom'] = [0, 1]
    #unrolling the second loop made no difference
    #tune_params['loop_unroll_factor_p1nlay'] = [0, 1, 3]

    adding_result = kt.run_kernel(adding_ref_kernel_name, ref_kernels_src, problem_size, adding_ref_kernel_args,
                                  dict(RTE_RRTMGP_USE_CBOOL=1, block_size_x=512, block_size_y=1), compiler_options=cp)

    adding_answer = [None for _ in adding_kernel_args]

    adding_answer[-6] = adding_result[-6]    # flux_up
    adding_answer[-5] = adding_result[-5]    # flux_dn
    adding_answer[-4] = adding_result[-4]    # flux_dir
    adding_answer[-3] = adding_result[-3]    # albedo
    adding_answer[-2] = adding_result[-2]    # src
    #adding_answer[-1] = adding_result[-1] # denom #removed

    print(f"Tuning {adding_kernel_name}")

    result, env = kt.tune_kernel(adding_kernel_name, kernels_src, problem_size, adding_kernel_args, tune_params, compiler_options=cp, answer=adding_answer,
                                 atol=1e-14, verbose=True, observers=[reg_observer], metrics=metrics, iterations=32, cache="adding_kernel_A100.json")

    with open('timings_sw_adding_kernel.json', 'w') as fp:
        json.dump(result, fp)


if __name__ == '__main__':
    command_line = parse_command_line()

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

    adding_kernel_args = [
        ncol, nlay, ngpt, top_at_1, sfc_alb_dif, r_dif, t_dif, source_dn, source_up, source_sfc, flux_up, flux_dn, flux_dir, albedo, src, denom
    ]

    problem_size = (ncol, ngpt)
    source_kernel_name = f'sw_source_kernel<{str_float}, {top_at_1}>'
    source_ref_kernel_name = f'sw_source_kernel<{str_float}>'
    adding_kernel_name = f'sw_adding_kernel<{str_float}, {top_at_1}>'
    adding_ref_kernel_name = f'sw_adding_kernel<{str_float}>'

    if command_line.tune:
        tune()
    elif command_line.run:
        parameters = dict()
        if command_line.best_configuration:
            with open('timings_sw_source_adding_kernel.json', 'r') as file:
                configurations = json.load(file)
            best_configuration = min(configurations, key=lambda x: x['time'])
            parameters['block_size_x'] = best_configuration['block_size_x']
            parameters['block_size_y'] = best_configuration['block_size_y']
        else:
            parameters['block_size_x'] = command_line.block_size_x
            parameters['block_size_y'] = command_line.block_size_y
        run_and_test(parameters)
