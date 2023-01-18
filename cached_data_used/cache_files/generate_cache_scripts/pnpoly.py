#!/usr/bin/env python
""" Point-in-Polygon host/device code tuner

This program is used for auto-tuning the host and device code of a CUDA program
for computing the point-in-polygon problem for very large datasets and large
polygons.

The time measurements used as a basis for tuning include the time spent on
data transfers between host and device memory. The host code uses device mapped
host memory to overlap communication between host and device with kernel
execution on the GPU. Because each input is read only once and each output
is written only once, this implementation almost fully overlaps all
communication and the kernel execution time dominates the total execution time.

The code has the option to precompute all polygon line slopes on the CPU and
reuse those results on the GPU, instead of recomputing them on the GPU all
the time. The time spent on precomputing these values on the CPU is also
taken into account by the time measurement in the code.

This code was written for use with the Kernel Tuner. See:
     https://github.com/benvanwerkhoven/kernel_tuner

Author: Ben van Werkhoven <b.vanwerkhoven@esciencecenter.nl>
"""
import sys
from collections import OrderedDict
import numpy as np
import kernel_tuner
import json
import logging

from kernel_tuner.nvml import NVMLObserver
from kernel_tuner.observers import CompiletimeObserver

import pycuda.driver as drv

def allocate(n, dtype=np.float32):
    """ allocate context-portable device mapped host memory """
    return drv.pagelocked_empty(int(n), dtype, order='C', mem_flags=drv.host_alloc_flags.PORTABLE|drv.host_alloc_flags.DEVICEMAP)

def tune(device_name, cc):

    #set the number of points and the number of vertices
    size = np.int32(2e7)
    problem_size = (size, 1)
    vertices = 600

    #allocate device mapped host memory and generate input data
    points = allocate(2*size, np.float32)
    np.copyto(points, np.random.randn(2*size).astype(np.float32))

    bitmap = allocate(size, np.int32)
    np.copyto(bitmap, np.zeros(size).astype(np.int32))
    #as test input we use a circle with radius 1 as polygon and
    #a large set of normally distributed points around 0,0
    vertex_seeds = np.sort(np.random.rand(vertices)*2.0*np.pi)[::-1]
    vertex_x = np.cos(vertex_seeds)
    vertex_y = np.sin(vertex_seeds)
    vertex_xy = allocate(2*vertices, np.float32)
    np.copyto(vertex_xy, np.array( list(zip(vertex_x, vertex_y)) ).astype(np.float32).ravel())

    #kernel arguments
    args = [bitmap, points, vertex_xy, size]

    #setup tunable parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,32)]  #multiple of 32
    tune_params["tile_size"] = [1] + [2*i for i in range(1,11)]
    tune_params["between_method"] = [0, 1, 2, 3]
    tune_params["use_precomputed_slopes"] = [0, 1]
    tune_params["use_method"] = [0, 1, 2]

    #tell the Kernel Tuner how to compute the grid dimensions from the problem_size
    grid_div_x = ["block_size_x", "tile_size"]

    metrics = OrderedDict()
    metrics["MPoints/s"] = lambda p : (size/1e6) / (p["time"]/1e3)

    #compute reference answer
    result = kernel_tuner.run_kernel("cn_pnpoly_naive", "pnpoly/pnpoly.cu", problem_size,
                                     [bitmap, points, size],
                                     {"block_size_x": 256}, cmem_args={"d_vertices": vertex_xy})
    reference = result[0].copy()
    answer = [reference, None, None, None]

    # setup observers
    nvmlobserver = NVMLObserver(["nvml_energy", "temperature", "core_freq", "mem_freq"])
    observers = [nvmlobserver, CompiletimeObserver()]

    #start tuning
    results = kernel_tuner.tune_kernel("cn_pnpoly_host", ['pnpoly/pnpoly_host.cu', 'pnpoly/pnpoly.cu'],
        problem_size, args, tune_params,
        grid_div_x=grid_div_x, lang="C", compiler_options=["-arch=sm_" + cc, "-Ipnpoly"], verbose=True,
        cache="pnpoly_" + device_name, metrics=metrics, iterations=32, observers=observers)

    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./GEMM.py [device name]")
        exit(1)
    device_name = sys.argv[1]

    drv.init()
    # context = drv.Device(1).make_context()
    context = drv.Device(0).make_context()

    #get compute capability for compiling CUDA kernels
    devprops = { str(k): v for (k, v) in context.get_device().get_attributes().items() }
    cc = str(devprops['COMPUTE_CAPABILITY_MAJOR']) + str(devprops['COMPUTE_CAPABILITY_MINOR'])

    try:
        tune(device_name, cc)
    finally:
        context.pop()

