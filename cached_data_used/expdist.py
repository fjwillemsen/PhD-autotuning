#!/usr/bin/env python
import sys
from collections import OrderedDict
import json
import numpy as np

from kernel_tuner import tune_kernel
from kernel_tuner.observers import BenchmarkObserver
from kernel_tuner.nvml import NVMLObserver


def tune(device_name, strategy="bayes_opt", strategy_options=None, verbose=True, quiet=False, simulation_mode=True):

    #setup tuning parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(8, 11)][::-1]    # 5
    tune_params["block_size_y"] = [2**i for i in range(6)]
    tune_params["tile_size_x"] = [1, 2, 3, 4, 5, 6, 7, 8]
    tune_params["tile_size_y"] = [1, 2, 3, 4, 5, 6, 7, 8]
    tune_params["use_shared_mem"] = [1]
    tune_params["loop_unroll_factor_x"] = [1, 2, 3, 4, 5, 6, 7, 8]
    tune_params["loop_unroll_factor_y"] = [1, 2, 3, 4, 5, 6, 7, 8]
    tune_params["use_column"] = [1]
    tune_params["use_separate_acc"] = [0]
    tune_params["n_y_blocks"] = [2**i for i in range(6)]

    def config_valid(p):
        if p["loop_unroll_factor_x"] > p["tile_size_x"] or (p["loop_unroll_factor_x"] and p["tile_size_x"] % p["loop_unroll_factor_x"] != 0):
            return False    #no need to test this loop unroll factor, as it is the same as not unrolling the loop
        if p["loop_unroll_factor_y"] > p["tile_size_y"] or (p["loop_unroll_factor_y"] and p["tile_size_y"] % p["loop_unroll_factor_y"] != 0):
            return False    #no need to test this loop unroll factor, as it is the same as not unrolling the loop
        return True

    restrictions = config_valid

    #setup test input
    alloc_size = 32 * 1024
    size = np.int32(32 * 1024)
    max_blocks = np.int32(np.ceil(size / float(np.amin(tune_params["block_size_x"]))) * np.ceil(size / float(np.amin(tune_params["block_size_y"]))))
    ndim = np.int32(2)
    A = np.random.randn(alloc_size * ndim).astype(np.float64)
    B = A + 0.00001 * np.random.randn(alloc_size * ndim).astype(np.float64)
    scale_A = np.absolute(0.01 * np.random.randn(alloc_size).astype(np.float64))
    scale_B = np.absolute(0.01 * np.random.randn(alloc_size).astype(np.float64))
    cost = np.zeros((max_blocks)).astype(np.float64)

    #setup kernel
    kernel_name = "ExpDist"
    arguments = [A, B, size, size, scale_A, scale_B, cost]
    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    #get number of registers
    class RegisterObserver(BenchmarkObserver):

        def get_results(self):
            return {
                "num_regs": self.dev.current_module.get_function(kernel_name).num_regs
            }

    problem_size = lambda p: (size, size if p["use_column"] == 0 else p["n_y_blocks"] * p["block_size_y"] * p["tile_size_y"])

    metrics = OrderedDict()
    metrics["registers"] = lambda p: p["num_regs"]
    metrics["clock"] = lambda p: p["core_freq"]

    def FLOPs_in_partial_reduction(p):
        num_thread_blocks = np.ceil(size / (p["block_size_x"] * p["tile_size_x"])) * np.ceil(size / (p["block_size_y"] * p["tile_size_y"]))
        ops_per_thread_block = p["block_size_x"] * p[
            "block_size_y"] / 32 * 31 + 31    #minimal number of ops per warp times number of warps + #ops for 1 final warp
        return num_thread_blocks * ops_per_thread_block

    ops_per_iteration = 35    #from Nsight profiler
    metrics["GFLOP/s"] = lambda p: ((FLOPs_in_partial_reduction(p) + ops_per_iteration * size * size) / 1e9) / (p["time"] / 1e3)

    cp = []

    kernel1, env = tune_kernel(kernel_name, "expdist.cu", problem_size, arguments, tune_params, grid_div_x=grid_div_x, grid_div_y=grid_div_y, metrics=metrics,
                               iterations=32, compiler_options=cp, cache="cache_files/expdist_" + device_name + "_processed", restrictions=restrictions,
                               strategy=strategy, strategy_options=strategy_options, simulation_mode=simulation_mode, verbose=verbose, quiet=quiet)
    return kernel1, env


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./expdist.py [device name]")
        exit(1)

    device_name = sys.argv[1]

    tune(device_name)
