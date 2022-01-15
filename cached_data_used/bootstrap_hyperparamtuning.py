#!/usr/bin/env python
import sys

import kernel_tuner
import numpy as np
from collections import OrderedDict

prog_name = "bootstrap_hyperparamtuning"


def tune(strategy="brute_force", strategy_options=None, verbose=False, quiet=False, simulation_mode=False):

    # input dimensions and data
    x = 100
    y = 100
    problem_size = (x, y)
    args = []
    metrics = OrderedDict()

    def make_dict(key: str, value):
        return { key: value }

    #setup tunable parameters
    tune_params = OrderedDict()
    tune_params["max_fevals"] = [220]
    tune_params["initialsamplemethod"] = ['lhs', 'index']
    tune_params["initialsamplerandomoffsetfactor"] = [0.01, 0.05, 0.1, 0.2, 0.3]
    tune_params["method"] = ['ei', 'poi']
    tune_params["methodparams"] = [ make_dict('explorationfactor', v) for v in [0.005, 0.01, 0.05, 0.1, 'CV']]
    tune_params["covariancekernel"] = ['matern', 'matern_scalekernel']
    tune_params["covariancelengthscale"] = [0.5, 1.5, 2.5]
    tune_params["likelihood"] = ['Gaussian', 'FixedNoise']    # TODO this causes a NanError for cholesky_cpu
    tune_params["optimizer"] = ['LBFGS', 'Adam']
    tune_params["optimizer_learningrate"] = [0.01, 0.1, 1]
    tune_params["initial_training_iter"] = [0, 10, 25, 50]
    tune_params["training_iter"] = [0, 1, 3]

    search_space_size = tuple(len(params) for params in tune_params.values())
    print(f"Search space size: {np.prod(search_space_size)}")

    strategy_options = {}
    strategy_options['max_fevals'] = 300

    #start tuning
    results, env = kernel_tuner.tune_kernel(prog_name + "_kernel", prog_name + "_kernel.py", problem_size, args, tune_params, lang='Python', iterations=1,
                                            metrics=metrics, verbose=verbose, quiet=quiet, strategy=strategy, strategy_options=strategy_options,
                                            simulation_mode=simulation_mode)

    # print(len(results))

    return results, env


if __name__ == "__main__":
    tune()
