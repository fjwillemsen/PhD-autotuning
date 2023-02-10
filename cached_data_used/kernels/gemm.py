#!/usr/bin/env python
import sys

from collections import OrderedDict
import os

import numpy as np
import kernel_tuner


def tune(device_name: str, strategy="mls", strategy_options=None, verbose=True, quiet=False, simulation_mode=True):

    path = os.path.dirname(os.path.realpath(__file__)) + "/gemm/"

    #// Matrices are accessed as follows:
    #// A: [k*M + m], with 'k' ranging from 0:K and 'm' from 0:M (m,k,m)
    #// B: [k*N + n], with 'k' ranging from 0:K and 'n' from 0:N (n,k,n)
    #// C: [n*M + m], with 'n' ranging from 0:N and 'm' from 0:M (m,n,m)
    n = np.int32(4096)
    m = np.int32(4096)
    k = np.int32(4096)

    A = np.array(np.random.randn(m, k), order='F').astype(np.float32)
    B = np.array(np.random.randn(k, n), order='F').astype(np.float32)
    C = np.zeros((m, n), order='F').astype(np.float32)

    alpha, beta = np.random.randn(2).astype(np.float32)
    alpha, beta = np.array([1.0, 1.0]).astype(np.float32)

    kernel_string = ""
    files = ["common.opencl", "xgemm_part1.opencl", "xgemm_part2.opencl", "xgemm_part3.opencl"]
    for f in files:
        with open(path + f, "r") as fp:
            kernel_string += fp.read()

    args = [m, n, k, alpha, beta, A, B, C]

    tune_params = OrderedDict()

    tune_params["MWG"] = [16, 32, 64, 128]
    tune_params["NWG"] = [16, 32, 64, 128]
    tune_params["KWG"] = [32]
    tune_params["MDIMC"] = [8, 16, 32]
    tune_params["NDIMC"] = [8, 16, 32]
    tune_params["MDIMA"] = [8, 16, 32]
    tune_params["NDIMB"] = [8, 16, 32]
    tune_params["KWI"] = [2]
    tune_params["VWM"] = [1, 2, 4, 8]
    tune_params["VWN"] = [1, 2, 4, 8]
    tune_params["STRM"] = [0]
    tune_params["STRN"] = [0]
    tune_params["SA"] = [0, 1]
    tune_params["SB"] = [0, 1]
    tune_params["PRECISION"] = [32]

    problem_size = (m, n)

    grid_div_x = ["MWG"]
    grid_div_y = ["NWG"]
    block_size_names = ["MDIMC", "NDIMC", "block_size_z"]

    restrict = []
    restrict += ["KWG % KWI == 0"]
    restrict += ["MWG % (MDIMC * VWM) == 0"]
    restrict += ["NWG % (NDIMC * VWN) == 0"]
    restrict += ["MWG % (MDIMA * VWM) == 0"]
    restrict += ["NWG % (NDIMB * VWN) == 0"]
    restrict += ["KWG % ((MDIMC * NDIMC)/MDIMA) == 0"]
    restrict += ["KWG % ((MDIMC * NDIMC)/NDIMB) == 0"]

    C_ref = (np.dot(alpha * A, B.T) + beta * C).astype(np.float32)
    answer = [None for _ in args]
    answer[-1] = C_ref

    metrics = OrderedDict()
    total_gflops = (float(m) * float(n) * k * 2.0 + 2.0 * float(m) * k) / 1e9
    metrics["GFLOP/s"] = lambda p: total_gflops / (p["time"] / 1e3)

    results, env = kernel_tuner.tune_kernel("Xgemm", kernel_string, problem_size, args, tune_params, block_size_names=block_size_names, lang="OpenCL",
                                            restrictions=restrict, compiler_options=["-I" + path], grid_div_x=grid_div_x, grid_div_y=grid_div_y, answer=answer,
                                            atol=1e-2, device=0, platform=0, metrics=metrics, iterations=32,
                                            cache="../cached_data_used/cachefiles/gemm/" + device_name.lower(), verbose=verbose, quiet=quiet, strategy=strategy,
                                            strategy_options=strategy_options, simulation_mode=simulation_mode)

    return results, env


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./GEMM.py [device name]")
        exit(1)

    device_name = sys.argv[1]

    tune(device_name)
