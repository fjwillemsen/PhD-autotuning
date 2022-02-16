#!/usr/bin/env python
from collections import OrderedDict
from time import perf_counter

from kernel_tuner import tune_kernel
from kernel_tuner import observers
from kernel_tuner.observers import BenchmarkObserver
import numpy as np

prog_name = "bootstrap_hyperparamtuning"


def tune(strategy="bayes_opt_GPyTorch_lean", iterations=35, strategy_options=None):
    simulation_mode = False
    parallel_mode = True
    verbose = False
    quiet = False

    # input dimensions and data
    x = 100
    y = 100
    problem_size = (x, y)
    args = []
    metrics = OrderedDict()
    strategy_options = {}
    strategy_options['max_fevals'] = 1000
    strategy_options['popsize'] = 50

    def make_dict(key: str, value):
        return {
            key: value
        }

    # # fixed tune params
    # tune_params = OrderedDict()
    # tune_params["max_fevals"] = [220]
    # tune_params["method"] = ['ei']
    # strategy_options['max_fevals'] = 1

    # # test precision
    # tune_params = OrderedDict()
    # tune_params["max_fevals"] = [220]
    # # tune_params["dummy_param"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # tune_params["precision"] = ['float', 'double']
    # tune_params["minimum_std"] = [1e-3, 1e-5, 1e-6, 1e-7, 1e-9]
    # strategy_options['max_fevals'] = np.prod(tuple(len(params) for params in tune_params.values()))

    # setup tunable parameters
    tune_params = OrderedDict()
    tune_params["max_fevals"] = [220]
    tune_params["initialsamplemethod"] = ['lhs', 'index']
    tune_params["initialsamplerandomoffsetfactor"] = [0.01, 0.05, 0.1, 0.2, 0.3]
    tune_params["method"] = ['ei', 'poi']
    tune_params["methodparams"] = [make_dict('explorationfactor', v) for v in [0.005, 0.01, 0.05, 0.1]]
    tune_params["covariancekernel"] = ['matern', 'matern_scalekernel']
    tune_params["covariancelengthscale"] = [0.5, 1.5, 2.5]
    # tune_params["likelihood"] = ['Gaussian', 'FixedNoise']    # TODO this causes a NanError for cholesky_cpu
    tune_params["optimizer"] = ['LBFGS', 'Adam', 'ASGD']
    tune_params["initial_training_iter"] = [5, 10, 25, 50]
    tune_params["training_iter"] = [0, 1, 3]

    search_space_size = tuple(len(params) for params in tune_params.values())
    print(f"Search space size: {np.prod(search_space_size)}")

    class DurationObserver(BenchmarkObserver):

        def before_start(self):
            super().before_start()
            self.start_time = perf_counter()

        def after_finish(self):
            super().after_finish()
            self.end_time = perf_counter()

        def get_results(self):
            super().get_results()
            return {
                'strategy_time': self.end_time - self.start_time
            }

    #start tuning
    results, env = tune_kernel(prog_name + "_kernel", prog_name + "_kernel.py", problem_size, args, tune_params, lang='Python', iterations=iterations,
                               metrics=metrics, verbose=verbose, quiet=quiet, strategy=strategy, strategy_options=strategy_options,
                               simulation_mode=simulation_mode, parallel_mode=parallel_mode, cache="cache_files/" + prog_name, observers=[])

    # print(len(results))

    return results, env


if __name__ == "__main__":
    tune()
