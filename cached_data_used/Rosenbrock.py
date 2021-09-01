#!/usr/bin/env python
import sys

import kernel_tuner
import numpy as np
from collections import OrderedDict

prog_name = "rosenbrock_constrained"


def tune(device_name, strategy="bayes_opt", strategy_options=None, verbose=False, quiet=False, simulation_mode=True):

    #input dimensions and data
    x = 100
    y = 100
    problem_size = None
    args = []
    metrics = OrderedDict()

    #setup tunable parameters
    tune_params = OrderedDict()
    tune_params["x"] = np.linspace(-1.5, 1.5, num=100).tolist()
    tune_params["y"] = np.linspace(-1.5, 1.5, num=100).tolist()

    #start tuning
    results, env = kernel_tuner.tune_kernel(prog_name + "_kernel", prog_name + ".cu", problem_size, args, tune_params, metrics=metrics, verbose=verbose,
                                            quiet=quiet, cache="cache_files/" + prog_name + "_" + device_name, strategy=strategy,
                                            strategy_options=strategy_options, simulation_mode=simulation_mode)

    # print(len(results))

    return results, env


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python {}.py [device name]".format(prog_name))
        exit(1)

    device_name = sys.argv[1]

    tune(device_name)
