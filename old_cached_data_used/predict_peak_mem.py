from collections import OrderedDict
from kernel_tuner import tune_kernel


def tune(device_name, strategy="bayes_opt_GPyTorch_lean", strategy_options=None, verbose=True, quiet=False, simulation_mode=True):

    num_layers = 42
    tune_params = OrderedDict()
    tune_params["gpu1"] = list(range(num_layers))
    tune_params["gpu2"] = list(range(num_layers))
    tune_params["gpu3"] = list(range(num_layers))
    tune_params["gpu4"] = list(range(num_layers))

    # each GPU must have at least one layer and the sum of all layers must not exceed the total number of layers
    # restrict = lambda p: min([p["gpu1"], p["gpu2"], p["gpu3"], p["gpu4"]]) >= 1 and sum([p["gpu1"], p["gpu2"], p["gpu3"], p["gpu4"]]) == num_layers

    from constraint import ExactSumConstraint, FunctionConstraint
    min_func = lambda gpu1, gpu2, gpu3, gpu4: min([gpu1, gpu2, gpu3, gpu4]) >= 1
    restrict = [ExactSumConstraint(num_layers), FunctionConstraint(min_func)]

    results, env = tune_kernel("predict_peak_mem", "predict_peak_mem_kernel.py", problem_size=1, iterations=1, arguments=[], tune_params=tune_params,
                               restrictions=restrict, verbose=verbose, quiet=quiet, cache=f"cache_files/predict_peak_mem_{device_name}", strategy=strategy,
                               strategy_options=strategy_options, lang="Python", simulation_mode=simulation_mode)

    return results, env

    # # Parameter space: a value between 1 and 100 indicating how many of the remaining layers are placed
    # # on a GPU (as a percentage).
    # opt = skopt.Optimizer([(1, 100), (1, 100), (1, 100)],
    #     "GP",
    #     n_initial_points=30,
    #     # acq_func="EI",
    #      acq_optimizer="sampling",
    #      acq_func_kwargs=acq_func_kwargs)

    # opt.run(predict_bo, n_iter=75)

    # print(opt.get_result())
    # print("Result:", opt.get_result().x, percent_to_layers(opt.get_result().x, global_n_layers), predict_bo(opt.get_result().x))
    # print("Absolute best AmoebaNet (predicted): [27, 45, 53] [11, 13, 9, 9] 6.392")


if __name__ == "__main__":
    strategy = "random_sample"
    strategy_options = {}
    strategy_options['max_fevals'] = 75
    strategy_options['popsize'] = 30
    tune("generator", strategy, strategy_options, None)
