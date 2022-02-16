#!/usr/bin/env python
import sys

import kernel_tuner
from kernel_tuner import hyper
import numpy as np
from collections import OrderedDict

prog_name = "mishras_bird_constrained"


def tune(device_name, strategy="bayes_opt", strategy_options=None, verbose=False, quiet=True, simulation_mode=True):

    #input dimensions and data
    x = 100
    y = 100
    problem_size = None
    args = []
    metrics = OrderedDict()

    #setup tunable parameters
    tune_params = OrderedDict()
    tune_params["x"] = np.linspace(-10, 0, num=100).tolist()
    tune_params["y"] = np.linspace(-6.5, 0, num=100).tolist()
    restrict = ["(x+5)**2 + (y+5)**2<25"]

    hyperparams = OrderedDict()
    # hyperparams['max_fevals'] = [100]
    hyperparams["af"] = ['poi', 'ei']
    hyperparams["samplingmethods"] = ['random', 'lhs']
    # hyperparams['samplingcriterion']: ['maximin', '']
    #                 'samplingiterations': 10000,

    #start tuning
    results = hyper.tune_hyper_params(strategy, hyperparams, prog_name + "_kernel", prog_name + ".cu", problem_size, args, tune_params, metrics=metrics,
                                      verbose=verbose, quiet=quiet, restrictions=restrict, cache="cache_files/" + prog_name + "_" + device_name,
                                      strategy_options=strategy_options, simulation_mode=simulation_mode)

    # print(len(results))

    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python {}.py [device name]".format(prog_name))
        exit(1)

    device_name = sys.argv[1]

    tune(device_name)
