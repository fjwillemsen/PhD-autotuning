from numpy.lib.function_base import disp
from caching import CacheInterface, CachedObject
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Tuple
from itertools import permutations
import statistics
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import math
import progressbar
import time as python_time

import importlib
import warnings
import os
import sys
import argparse


def change_dir(dir: str):
    os.chdir(dir)
    sys.path.append(dir)


def generate_strategies(variation_type: str, ignore_cache=False, default_number_of_repeats=7,
                        default_number_of_evaluations=list(np.array(range(20, 221, 20)).astype(int))) -> dict:
    """ Generates the setup of the strategies (beware that the options determine the maximum number of iterations, so setting this lower than the num_of_evaluations causes problems) """
    max_num_evals = max(default_number_of_evaluations)
    strategy_name = 'bayes_opt'
    variation_types = [
        'acquisition_functions', 'acquisition_functions_exploration', 'acquisition_functions_multi', 'covariance_function', 'initial_sampling',
        'multi_af_order', 'multi_af_discount_factor', 'multi_af_required_improvement_factor'
    ]
    if variation_type not in variation_types:
        raise ValueError(f"{variation_type} not in {variation_types}")
    acquisition_functions = ['ei', 'poi', 'lcb', 'multi', 'multi-advanced-precise']
    acquisition_functions_display_names = ['EI', 'PoI', 'LCB', 'multi', 'multi-advanced-precise']
    # acquisition_functions = ['multi', 'multi-fast', 'multi-advanced-precise']
    # acquisition_functions_display_names = ['Multi', 'Naive Multi', 'Advanced Multi']
    # acquisition_functions = ['multi-advanced-precise']
    # acquisition_functions_display_names = ['Advanced Multi']
    # variations = ['CV']
    # variations = [0.01, 0.1, 0.3, 'CV']
    # variations = [5, 10, 15, 20, 25, 30]
    # variations = ['matern32', 'matern52']
    variations = ['random', None, 'correlation', 'maximin', 'ratio']
    # variations = list(permutations(['ei', 'poi', 'lcb'], 3))
    # variations = [0.03, 0.05, 0.1, 0.15, 0.2, 0.25]
    # variations = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    append_key = 'IS'
    group_by_variation = len(variations) > 1
    variation_in_display_name = False
    reference_strategies = {
        'random_sample': {
            'name': 'random_sample',
            'strategy': 'random_sample',
            'display_name': 'Random Sample',
            'bar_group': 'reference',
            'nums_of_evaluations': default_number_of_evaluations,
            'repeats': 100,
            'options': {
                'fraction': 0.05
            }
        },
        'genetic_algorithm': {
            'name': 'genetic_algorithm',
            'strategy': 'genetic_algorithm',
            'display_name': 'Genetic Algorithm',
            'bar_group': 'reference',
            'nums_of_evaluations': default_number_of_evaluations,
            'repeats': 21,
            'options': {
                'max_fevals': max_num_evals,
            }
        },
    }
    strategies = reference_strategies
    for af_index, af in enumerate(acquisition_functions):
        for variation in variations:
            key = f"{strategy_name}_{af}_{append_key}{variation}"
            af_display_name = acquisition_functions_display_names[af_index]
            display_name = f"{af_display_name} {variation}" if variation_in_display_name else af_display_name
            # add defaults
            dct = dict()
            dct['name'] = key
            dct['strategy'] = strategy_name
            dct['display_name'] = display_name
            dct['bar_group'] = str(variation) if group_by_variation else af_display_name
            dct['nums_of_evaluations'] = default_number_of_evaluations
            dct['repeats'] = default_number_of_repeats
            if ignore_cache is True:
                dct['ignore_cache'] = True
            # add options
            options = dict()
            options['max_fevals'] = max_num_evals
            options['method'] = af
            if variation_type == 'acquisition_functions_exploration':
                options['methodparams'] = {
                    'explorationfactor': variation,
                    'zeta': 1,
                    'skip_duplicate_after': 30,
                }
            if variation_type == 'acquisition_functions_multi':
                options['methodparams'] = {
                    'explorationfactor': 'CV',
                    'zeta': 1,
                    'skip_duplicate_after': variation,
                }
            if variation_type == 'covariance_function':
                options['covariancekernel'] = variation
            if variation_type == 'multi_af_order':
                options['multi_af_names'] = list(variation)
            if variation_type == 'multi_af_discount_factor':
                options['multi_af_discount_factor'] = variation
            if variation_type == 'multi_af_required_improvement_factor':
                options['multi_afs_required_improvement_factor'] = variation
            if variation_type == 'initial_sampling':
                if variation == 'random':
                    options['samplingmethod'] = 'random'
                else:
                    options['samplingcriterion'] = variation
            dct['options'] = options
            # write to strategies
            strategies[key] = dct
    return strategies


def calculate_lower_upper_error(observations: list) -> Tuple[float, float]:
    """ Calculate the lower and upper error by the mean of the values below and above the median respectively """
    observations.sort()
    middle_index = len(observations) // 2
    middle_index_upper = middle_index + 1 if len(observations) % 2 != 0 else middle_index
    lower_values = observations[:middle_index]
    upper_values = observations[middle_index_upper:]
    lower_error = np.mean(lower_values)
    upper_error = np.mean(upper_values)
    return lower_error, upper_error


class StatisticalData():
    """ Object that captures all statistical data and functions to experiment and visualize, plots have possible metrics 'GFLOP/s' or 'time' """

    x_metric_displayname = dict({
        'num_evals': 'Number of function evaluations used',
        'strategy_time': 'Average time taken by strategy in miliseconds',
        'compile_time': 'Average compile time in miliseconds',
        'execution_time': 'Evaluation execution time taken in miliseconds',
        'total_time': 'Average total time taken in miliseconds',
    })

    y_metric_displayname = dict({
        'time': 'Best found time in miliseconds',
        'GFLOP/s': 'GFLOP/s',
    })

    def __init__(self, kernels: list, kernel_names: list, device_names: list, absolute_optima: list, kernel_defaults: dict, simulation_mode=True):
        self.kernel = kernels
        self.kernel_names = kernel_names
        self.absolute_optima = absolute_optima
        self.kernel_defaults = kernel_defaults
        self.device_names = device_names
        self.simulation_mode = simulation_mode

        # print(generate_strategies(variation_type='initial_sampling'))
        # exit(0)

        # What is the optimal selected acquisition function?
        compare_selected_acquisition_functions = {
            'random_sample': {
                'name': 'random_sample',
                'strategy': 'random_sample',
                'display_name': 'Random Sample',
                'bar_group': 'reference',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 100,
                'options': {
                    'fraction': 0.05
                }
            },
            'bayes_opt_ei_CV_reference': {
                'name': 'bayes_opt_ei_CV_reference',
                'strategy': 'bayes_opt',
                'display_name': 'EI',
                'bar_group': 'basic',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220,
                    'method': 'ei',
                }
            },
            'bayes_opt_multi_reference': {
                'name': 'bayes_opt_multi_reference',
                'strategy': 'bayes_opt',
                'display_name': 'Multi',
                'bar_group': 'multi',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220,
                    'method': 'multi',
                }
            },
            'bayes_opt_multi-advanced_reference': {
                'name': 'bayes_opt_multi-advanced_reference',
                'strategy': 'bayes_opt',
                'display_name': 'Advanced Multi',
                'bar_group': 'multi',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220,
                    'method': 'multi-advanced',
                }
            },
        }

        # What is the optimal search method?
        # Basinhopping is not used because it performs very poorly and does not work on the adding kernel
        compare_search_method = {
            'random_sample': {
                'name': 'random_sample',
                'strategy': 'random_sample',
                'display_name': 'Random',
                'bar_group': 'reference',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 100,
                'options': {
                    'fraction': 0.05
                }
            },
            'simulated_annealing': {
                'name': 'simulated_annealing',
                'strategy': 'simulated_annealing',
                'display_name': 'SA',
                'bar_group': 'reference',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220
                }
            },
            'mls': {
                'name': 'mls',
                'strategy': 'mls',
                'display_name': 'MLS',
                'bar_group': 'reference',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220,
                }
            },
            'genetic_algorithm': {
                'name': 'genetic_algorithm',
                'strategy': 'genetic_algorithm',
                'display_name': 'GA',
                'bar_group': 'reference',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220
                }
            },
            'bayes_opt_ei_CV_reference': {
                'name': 'bayes_opt_ei_CV_reference',
                'strategy': 'bayes_opt',
                'display_name': 'EI',
                'bar_group': 'bo',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220,
                    'method': 'ei',
                }
            },
            'bayes_opt_multi_reference': {
                'name': 'bayes_opt_multi_reference',
                'strategy': 'bayes_opt',
                'display_name': 'Multi',
                'bar_group': 'bo',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220,
                    'method': 'multi',
                }
            },
            'bayes_opt_multi-advanced_reference': {
                'name': 'bayes_opt_multi-advanced_reference',
                'strategy': 'bayes_opt',
                'display_name': 'Advanced Multi',
                'bar_group': 'bo',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220,
                    'method': 'multi-advanced',
                }
            },
        }

        # What is the optimal long-term search method?
        # Basinhopping is not used because it performs very poorly and does not work on the adding kernel
        compare_search_method_extended = {
            'extended_random_sample_rem_unique': {
                'name':
                'extended_random_sample_rem_unique',
                'strategy':
                'random_sample',
                'display_name':
                'Random',
                'bar_group':
                'reference',
                'nums_of_evaluations': [
                    20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580,
                    600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
                ],
                'repeats':
                100,
                'options': {
                    'fraction': 0.5
                }
            },
            'extended_simulated_annealing_rem_unique': {
                'name':
                'extended_simulated_annealing_rem_unique',
                'strategy':
                'simulated_annealing',
                'display_name':
                'SA',
                'bar_group':
                'reference',
                'nums_of_evaluations': [
                    20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580,
                    600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
                ],
                'repeats':
                35,
                'options': {
                    'max_fevals': 1020
                }
            },
            'extended_mls_rem_unique': {
                'name':
                'extended_mls_rem_unique',
                'strategy':
                'mls',
                'display_name':
                'MLS',
                'bar_group':
                'reference',
                'nums_of_evaluations': [
                    20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580,
                    600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
                ],
                'repeats':
                35,
                'options': {
                    'max_fevals': 1020,
                }
            },
            'extended_genetic_algorithm_rem_unique': {
                'name':
                'extended_genetic_algorithm_rem_unique',
                'strategy':
                'genetic_algorithm',
                'display_name':
                'GA',
                'bar_group':
                'reference',
                'nums_of_evaluations': [
                    20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580,
                    600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980, 1000, 1020
                ],
                'repeats':
                35,
                'options': {
                    'max_fevals': 1020,
                    'maxiter': 200,
                    'popsize': 80,
                }
            },
            'bayes_opt_multi_reference': {
                'name': 'bayes_opt_multi_reference',
                'strategy': 'bayes_opt',
                'display_name': 'Multi',
                'bar_group': 'multi',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220,
                    'method': 'multi',
                }
            },
            'bayes_opt_multi-advanced_reference': {
                'name': 'bayes_opt_multi-advanced_reference',
                'strategy': 'bayes_opt',
                'display_name': 'Advanced Multi',
                'bar_group': 'bo',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 35,
                'options': {
                    'max_fevals': 220,
                    'method': 'multi-advanced',
                }
            },
        }
        self.strategies = compare_search_method

        # # adds ignore_cache to every strategy for a complete rerun
        # for key in self.strategies.keys():
        #     if self.strategies[key]['bar_group'] == 'reference':
        #         self.strategies[key]['ignore_cache'] = True

        self.caches = list()
        for index, kernel in enumerate(kernels):
            if len(kernels) > 1:
                print()
                print(f"-|-|- Kernel {kernel_names[index]} on {device_names[index]} -|-|-")
            self.caches.append(self.collect_data(kernel, kernel_names[index], device_names[index]))

    def create_results_dict(self, strategy: dict) -> dict:
        """ Creates a results dictionary for this strategy """
        if len(strategy['nums_of_evaluations']) <= 0:
            return None

        number_of_evaluations = np.array(strategy['nums_of_evaluations']).astype(int)
        number_of_evaluations_as_keys = number_of_evaluations.astype(str).tolist()

        # fill the results with the default values
        results_per_number_of_evaluations_stats = {
            'actual_num_evals': np.array([]),
            'time': np.array([]),
            'GFLOP/s': np.array([]),
            'cumulative_execution_time': np.array([]),
            'mean_actual_num_evals': 0,
            'mean_GFLOP/s': 0,
            'mean_time': 0,
            'err_actual_num_evals': 0,
            'err_GFLOP/s': 0,
            'err_time': 0,
            'mean_cumulative_strategy_time': 0,
            'mean_cumulative_compile_time': 0,
            'mean_cumulative_execution_time': 0,
            'mean_cumulative_total_time': 0,
        }
        expected_results = dict({
            'results_per_number_of_evaluations': dict.fromkeys(number_of_evaluations_as_keys),
        })
        for num_of_evaluations in number_of_evaluations_as_keys:
            expected_results['results_per_number_of_evaluations'][num_of_evaluations] = deepcopy(results_per_number_of_evaluations_stats)
        return expected_results

    def tune(self, kernel, kernel_name, device_name, strategy) -> Tuple[list, int]:
        """ Execute a strategy """
        total_start_time = python_time.perf_counter()
        warnings.simplefilter("ignore", UserWarning)
        try:
            res, _ = kernel.tune(device_name=device_name, strategy=strategy['strategy'], strategy_options=strategy['options'], verbose=False, quiet=True,
                                 simulation_mode=self.simulation_mode)
        except ValueError:
            print(f"Something went wrong, trying once more.")
            res, _ = kernel.tune(device_name=device_name, strategy=strategy['strategy'], strategy_options=strategy['options'], verbose=False, quiet=True,
                                 simulation_mode=self.simulation_mode)
        warnings.simplefilter("default", UserWarning)
        total_end_time = python_time.perf_counter()
        total_time_ms = round((total_end_time - total_start_time) * 1000)
        return res, total_time_ms

    def collect_data(self, kernel, kernel_name, device_name, optimization='time', remove_duplicate_results=True) -> CachedObject:
        """ Executes strategies to obtain (or retrieve from cache) the statistical data """
        # get or create a cache
        cache = CachedObject(kernel_name, device_name, deepcopy(self.strategies))

        # run all strategies
        for strategy in self.strategies.values():
            print(f"Running {strategy['display_name']} {strategy['bar_group'] if 'bar_group' in strategy else ''}")
            nums_of_evaluations = strategy['nums_of_evaluations']
            self.min_num_evals = min(strategy['nums_of_evaluations'])
            self.max_num_evals = max(strategy['nums_of_evaluations'])

            # if the strategy is in the cache, use cached data
            if 'ignore_cache' not in strategy:
                expected_results = self.create_results_dict(strategy)
                cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                if cached_data is not None:
                    print("| retrieved from cache")
                    continue

            def remove_duplicates(res):
                if not remove_duplicate_results:
                    return res
                unique_res = list()
                for result in res:
                    if result not in unique_res:
                        unique_res.append(result)
                return unique_res

            # repeat the strategy as specified
            repeated_results = list()
            total_time_results = np.array([])
            for rep in progressbar.progressbar(range(strategy['repeats']), redirect_stdout=True):
                # print(rep + 1, end=', ', flush=True)
                res, total_time_ms = self.tune(kernel, kernel_name, device_name, strategy)
                # check if there are only invalid configs
                only_invalid = len(res) < 1 or min(res[:20], key=lambda x: x['time'])['time'] == 1e20
                unique_res = remove_duplicates(res)
                while only_invalid or (remove_duplicate_results and len(unique_res) < self.max_num_evals):
                    if len(res) < 1:
                        print(f"({rep+1}/{strategy['repeats']}) No results found, trying once more...")
                    elif len(unique_res) < self.max_num_evals:
                        print(f"Too few unique results found ({len(unique_res)} in {len(res)} evaluations), trying once more...")
                    else:
                        print(f"({rep+1}/{strategy['repeats']}) Only invalid results found, trying once more...")
                    res, total_time_ms = self.tune(kernel, kernel_name, device_name, strategy)
                    only_invalid = len(res) < 1 or min(res[:20], key=lambda x: x['time'])['time'] == 1e20
                    unique_res = remove_duplicates(res)
                # register the results
                repeated_results.append(unique_res)
                total_time_results = np.append(total_time_results, total_time_ms)
                if len(strategy['nums_of_evaluations']) <= 0:
                    nums_of_evaluations = np.append(nums_of_evaluations, len(unique_res))

            # create the results dict for this strategy
            strategy['nums_of_evaluations'] = nums_of_evaluations
            results_to_write = self.create_results_dict(strategy)
            total_time_mean = np.mean(total_time_results)
            total_time_std = np.std(total_time_results)
            total_time_mean_per_eval = total_time_mean / nums_of_evaluations[-1]
            print("Total mean time: {} ms, std {}".format(round(total_time_mean, 3), round(total_time_std, 3)))

            # for every number of evaluations specified, find the best and collect details on it
            for res_index, res in enumerate(repeated_results):
                for num_of_evaluations in nums_of_evaluations:
                    limited_res = res[:num_of_evaluations]
                    if optimization == 'time':
                        best = min(limited_res, key=lambda x: x['time'])
                    elif optimization == 'GFLOP/s':
                        best = max(limited_res, key=lambda x: x['GFLOP/s'])
                    time = best['time']
                    if time == 1e20:
                        error_message = f"({res_index+1}/{len(repeated_results)}) Only invalid values found after {num_of_evaluations} evaluations for strategy {strategy['display_name']}. Values: {limited_res}"
                        raise ValueError(error_message)
                    gflops = best['GFLOP/s'] if 'GFLOP/s' in best else np.nan
                    cumulative_execution_time = sum(x['time'] for x in limited_res if x['time'] != 1e20)

                    # write to the results
                    result = results_to_write['results_per_number_of_evaluations'][str(num_of_evaluations)]
                    result['actual_num_evals'] = np.append(result['actual_num_evals'], len(limited_res))
                    result['time'] = np.append(result['time'], time)
                    result['GFLOP/s'] = np.append(result['GFLOP/s'], gflops)
                    result['cumulative_execution_time'] = np.append(result['cumulative_execution_time'], cumulative_execution_time)

            # check and summarise results
            for num_of_evaluations in nums_of_evaluations:
                result = results_to_write['results_per_number_of_evaluations'][str(num_of_evaluations)]
                result['mean_actual_num_evals'] = np.mean(result['actual_num_evals'])
                result['mean_time'] = np.mean(result['time'])
                result['mean_GFLOP/s'] = np.mean(result['GFLOP/s'])
                result['err_actual_num_evals'] = np.std(result['actual_num_evals'])
                result['err_time'] = np.std(result['time'])
                result['err_GFLOP/s'] = np.std(result['GFLOP/s'])
                result['mean_cumulative_compile_time'] = 0
                result['mean_cumulative_execution_time'] = np.mean(result['cumulative_execution_time'])
                cumulative_total_time = total_time_mean_per_eval * num_of_evaluations
                if self.simulation_mode:
                    result['mean_cumulative_strategy_time'] = cumulative_total_time
                    result['mean_cumulative_total_time'] = cumulative_total_time + result['mean_cumulative_compile_time'] + result[
                        'mean_cumulative_execution_time']
                else:
                    result['mean_cumulative_strategy_time'] = cumulative_total_time - result['mean_cumulative_compile_time'] - result[
                        'mean_cumulative_execution_time']
                    result['mean_cumulative_total_time'] = cumulative_total_time
                # if result['err_actual_num_evals'] > 0:
                #     raise ValueError('The number of actual evaluations over the runs has varied: {}'.format(result['actual_num_evals']))
                # if result['mean_actual_num_evals'] != num_of_evaluations:
                #     print(
                #         "The set number of evaluations ({}) is not equal to the actual number of evaluations ({}). Try increasing the fraction or maxiter in strategy options."
                #         .format(num_of_evaluations, result['mean_actual_num_evals']))

            # write to the cache
            cache.set_strategy(deepcopy(strategy), results_to_write)
            # print("")
        return cache

    def order_strategies(self, main_bar_group: str) -> list:
        """ Orders the strategies in the order we wish to have them plotted """
        strategies_ordered = [None] * len(self.strategies.values())
        index_first = 0
        index_last = len(strategies_ordered) - 1
        for strategy in self.strategies.values():
            if 'bar_group' not in strategy:
                strategies_ordered = self.strategies.values()
                break
            if strategy['bar_group'] == main_bar_group:
                strategies_ordered[index_last] = strategy
                index_last -= 1
            else:
                strategies_ordered[index_first] = strategy
                index_first += 1
        if main_bar_group == '' or None in strategies_ordered:
            strategies_ordered = self.strategies.values()
        return strategies_ordered

    def plot_strategies_errorbar(self, kernel_index=0, x_metric='num_evals', y_metric='GFLOP/s', shaded=True, plot_errors=True, main_bar_group=''):
        """ Plots all strategies with errorbars, shaded plots a shaded error region instead of error bars. Y-axis and X-axis metrics can be chosen. """
        # if len(self.caches) != 1:
        #     raise ValueError("This function does not support plotting multiple kernels.")
        cache = self.caches[kernel_index]

        # plotting setup
        bar_groups_markers = {
            'reference': '.',
            'basic': '+',
            'multi': 'd',
            'default': '+',
            'pruned': 'd',
            'bo': 'D',
        }
        kernel_name = self.kernel_names[kernel_index]
        figname = kernel_name
        fig = plt.figure(figname)
        ax = fig.add_subplot(111)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        strategies_ordered = self.order_strategies(main_bar_group)

        # plot absolute optimum
        absolute_optimum = self.absolute_optima[kernel_index]
        if absolute_optimum is not None:
            ax.plot([self.min_num_evals, self.max_num_evals], [absolute_optimum, absolute_optimum], linestyle='-',
                    label="True optimum {}".format(round(absolute_optimum, 3)), color='black')

        # plotting
        for strategy_index, strategy in enumerate(strategies_ordered):
            color = colors[strategy_index]
            collect_xticks = list()
            # must use cached data written by collect_data() earlier
            expected_results = self.create_results_dict(strategy)
            cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
            if cached_data is not None:
                results = cached_data['results']['results_per_number_of_evaluations']
                perf = np.array([])
                perf_error = np.array([])
                perf_error_lower = np.array([])
                perf_error_upper = np.array([])
                actual_num_evals = np.array([])
                cumulative_strategy_time = np.array([])
                cumulative_compile_time = np.array([])
                cumulative_execution_time = np.array([])
                cumulative_total_time = np.array([])
                collect_xticks.append(list(int(x) for x in results.keys()))
                for key in results.keys():
                    result = results[key]
                    # calculate y axis data
                    perf = np.append(perf, result['mean_' + y_metric])
                    perf_error = np.append(perf_error, result['err_' + y_metric])
                    # calculate the lower and upper error by dividing the observations (without the middle value) and taking the means
                    observations = result[y_metric]
                    lower_error, upper_error = calculate_lower_upper_error(observations)
                    # print(f"{key} | {result['mean_' + y_metric]}: ml {lower_error} mh {upper_error} | {lower_values} {upper_values}")
                    perf_error_lower = np.append(perf_error_lower, lower_error)
                    perf_error_upper = np.append(perf_error_upper, upper_error)
                    # calculate x axis data
                    # mean_actual_num_evals are the mean number of evaluations without invalid configurations, but we want to include the invalid configurations in the number of evaluations for a more complete result
                    # actual_num_evals = np.append(actual_num_evals, result['mean_actual_num_evals'])
                    actual_num_evals = np.append(actual_num_evals, int(key))
                    cumulative_strategy_time = np.append(cumulative_strategy_time, result['mean_cumulative_strategy_time'])
                    cumulative_compile_time = np.append(cumulative_compile_time, result['mean_cumulative_compile_time'])
                    cumulative_execution_time = np.append(cumulative_execution_time, result['mean_cumulative_execution_time'])
                    cumulative_total_time = np.append(cumulative_total_time, result['mean_cumulative_total_time'])
                # set x axis data
                if x_metric == 'num_evals':
                    x_axis = actual_num_evals
                elif x_metric == 'strategy_time':
                    x_axis = cumulative_strategy_time
                elif x_metric == 'compile_time':
                    x_axis = cumulative_compile_time
                elif x_metric == 'execution_time':
                    x_axis = cumulative_execution_time
                elif x_metric == 'total_time':
                    x_axis = cumulative_total_time
                else:
                    raise ValueError("Invalid x-axis metric")
                # plot and add standard deviation to the plot
                marker = '0'
                alpha = 1.0
                fill_alpha = 0.2
                plot_error = plot_errors
                if 'bar_group' in strategy:
                    bar_group = strategy['bar_group']
                    marker = bar_groups_markers[bar_group]
                    if bar_group == 'reference':
                        fill_alpha = 0.2
                        plot_error = plot_errors
                    elif main_bar_group != '' and bar_group != main_bar_group:
                        alpha = 0.7
                        fill_alpha = 0.0
                        plot_error = False
                    if main_bar_group == 'bo' and bar_group == main_bar_group:
                        alpha = 1.0
                        fill_alpha = 0.0
                        plot_error = False
                if shaded is True:
                    if plot_error:
                        ax.fill_between(x_axis, perf_error_lower, perf_error_upper, alpha=fill_alpha, antialiased=True, color=color)
                    ax.plot(x_axis, perf, marker=marker, alpha=alpha, linestyle='--', label=strategy['display_name'], color=color)
                else:
                    ax.errorbar(x_axis, perf, perf_error, marker=marker, alpha=alpha, linestyle='--', label=strategy['display_name'])
            else:
                raise ValueError("Strategy {} not in cache, make sure collect_data() has ran first".format(strategy['display_name']))

        # set the y-axis limit
        y_axis_lower_limit = absolute_optimum - ((ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01)
        ax.set_ylim(y_axis_lower_limit, ax.get_ylim()[1])
        kernel_defaults = self.kernel_defaults[kernel_name]
        if 'y_axis_upper_limit' in kernel_defaults:
            y_axis_upper_limit = kernel_defaults['y_axis_upper_limit']
            if y_axis_upper_limit is not None:
                y_axis_lower_limit = absolute_optimum - ((y_axis_upper_limit - ax.get_ylim()[0]) * 0.01)
                ax.set_ylim(y_axis_lower_limit, y_axis_upper_limit)

        # plot setup
        if len(collect_xticks) > 0:
            ax.set_xticks(collect_xticks[0])
        ax.set_xlabel(self.x_metric_displayname[x_metric])
        ax.set_ylabel(self.y_metric_displayname[y_metric])
        ax.legend()
        if plot_error is False:
            ax.grid(axis='y', zorder=0, alpha=0.7)
        fig.tight_layout()
        plt.show()

    def plot_RMSE_barchart(self, kernel_indices=[0], y_metric='time', mean_absolute_error=True, error_bars=True, bar_width=0.1, swap_groups_names=False,
                           displayname_as_group=False, ref_value_in_legend=False, log_scale=True, log_output=False):
        """ Root Mean Squared Error barchart, skipping the first set of evaluations to avoid weirdness. Can compare multiple kernels if they have the same range of results. """
        if log_output:
            print(f"RMSE, metric {self.y_metric_displayname[y_metric]}:")
        groups = defaultdict(list)
        groups_lower_err = defaultdict(list)
        groups_upper_err = defaultdict(list)
        ref_names = list()
        names = list()
        for strategy in self.strategies.values():
            for kernel_index in kernel_indices:
                absolute_optimum = self.absolute_optima[kernel_index]
                # must use cached data written by collect_data() earlier
                expected_results = self.create_results_dict(strategy)
                cache = self.caches[kernel_index]
                cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                if cached_data is None:
                    raise ValueError("Strategy {} not in cache, make sure collect_data() has ran first".format(strategy['display_name']))
                results = cached_data['results']['results_per_number_of_evaluations']

                # Calculate the MAE / RMSE values and lower and upper errors
                counter = 0
                runs = results[list(results.keys())[0]][y_metric]
                RMSE_sums = [0] * len(runs)
                for index, key in enumerate(results.keys()):
                    # skip the first 20 iterations because it is relatively noisy and dependent on initial sample
                    if index < 1:
                        continue
                    counter += 1
                    result = results[key]
                    # calculate the seperate MAE sums
                    for index, res in enumerate(result[y_metric]):
                        diff = np.abs(res - absolute_optimum)
                        if not mean_absolute_error:
                            diff = diff**2
                        RMSE_sums[index] += diff
                RMSEs = list()
                for index, _ in enumerate(runs):
                    RMSEs.append(RMSE_sums[index] / counter)
                rmse = np.mean(RMSEs)
                rmse_lower_err, rmse_upper_err = calculate_lower_upper_error(RMSEs)

                if log_output:
                    print(f"{strategy['display_name']}: {rmse}")
                # add the value to the group
                if len(kernel_indices) > 1 or displayname_as_group:
                    group = strategy['display_name']
                else:
                    group = strategy['bar_group'] if 'bar_group' in strategy else 'ignore'
                # add the name to the xtick list
                if len(kernel_indices) == 1:
                    name = strategy['display_name']
                else:
                    name = self.kernel_names[kernel_index]
                if group != 'reference' and swap_groups_names:
                    temp = name
                    name = group
                    group = temp
                # register in groups
                groups[group].append(rmse)
                groups_lower_err[group].append(rmse_lower_err)
                groups_upper_err[group].append(rmse_upper_err)
                # register names
                if group == 'reference' and name:
                    ref_names.append(name)
                else:
                    if name not in names and name:
                        names.append(name)
        # plotting
        if len(kernel_indices) == 1:
            figname = self.kernel_names[kernel_indices[0]]
        else:
            figname = f"plot"
        fig = plt.figure(figname)
        ax = fig.add_subplot(111)
        index = np.array([0])
        offset = -bar_width
        # plot the barchart groups
        for group in groups.keys():
            if group == 'reference':
                continue
            offset += bar_width
            index = np.arange(len(groups[group])) + offset
            label = group if group != 'ignore' else None
            if error_bars is True:
                ax.bar(index, groups[group], yerr=(groups_lower_err[group], groups_upper_err[group]), width=bar_width, zorder=5, label=label, alpha=0.8,
                       error_kw=dict(lw=1, capsize=2))
            else:
                ax.bar(index, groups[group], width=bar_width, zorder=5, label=label, alpha=0.8)
        # plot the reference lines
        for i, ref in enumerate(groups['reference']):
            label = f"{ref_names[i]} ({round(ref, 3)})" if ref_value_in_legend else ref_names[i]
            # offset_ref = bar_width if len(index) <= 1 else offset
            # x_axis = [min(index) - offset_ref, max(index) + offset_ref]
            padding = bar_width * 1
            # padding = 0
            x_axis = (ax.get_xlim()[0] + padding, ax.get_xlim()[1] - padding)
            ax.plot(x_axis, [ref, ref], linestyle='--', label=label, zorder=0)
        ax.set_ylabel('MAE' if mean_absolute_error else 'RMSE')
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylim(10e-3, ax.get_ylim()[1])
        ax.set_xticks(index - offset / 2)
        ax.set_xticklabels(names)
        ax.grid(axis='y', zorder=0, alpha=0.7)
        ax.legend(loc='upper center')
        # ax.legend(loc='upper right')
        # ax.legend()
        fig.tight_layout()
        plt.show()

    def plot_strategies_timebarchart(self):
        """ Plots each strategy in a separate bar chart plot """

        # plt.subplots()

        for strategy in self.strategies.values():

            # must use cached data written by collect_data() earlier
            expected_results = self.create_results_dict(strategy)
            cached_data = self.cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
            if cached_data is not None:
                results = cached_data['results']['results_per_number_of_evaluations']
                perf = np.array([])
                perf_error = np.array([])
                actual_num_evals = np.array([])
                cumulative_strategy_time = np.array([])
                cumulative_compile_time = np.array([])
                cumulative_execution_time = np.array([])
                cumulative_total_time = np.array([])
                for key in results.keys():
                    result = results[key]
                    # calculate y axis data
                    perf = np.append(perf, result['mean_' + y_metric])
                    perf_error = np.append(perf_error, result['err_' + y_metric])
                    # calculate x axis data
                    actual_num_evals = np.append(actual_num_evals, result['mean_actual_num_evals'])
                    cumulative_strategy_time = np.append(cumulative_strategy_time, result['mean_cumulative_strategy_time'])
                    cumulative_compile_time = np.append(cumulative_compile_time, result['mean_cumulative_compile_time'])
                    cumulative_execution_time = np.append(cumulative_execution_time, result['mean_cumulative_execution_time'])
                    cumulative_total_time = np.append(cumulative_total_time, result['mean_cumulative_total_time'])
                # set x axis data
                if x_metric == 'num_evals':
                    x_axis = actual_num_evals
                elif x_metric == 'strategy_time':
                    x_axis = cumulative_strategy_time
                elif x_metric == 'compile_time':
                    x_axis = cumulative_compile_time
                elif x_metric == 'execution_time':
                    x_axis = cumulative_execution_time
                elif x_metric == 'total_time':
                    x_axis = cumulative_total_time
                else:
                    raise ValueError("Invalid x-axis metric")
                # plot and add standard deviation to the plot
                if shaded:
                    plt.plot(x_axis, perf, marker='o', linestyle='--', label=strategy['display_name'])
                    plt.fill_between(x_axis, perf - perf_error, perf + perf_error, alpha=0.2, antialiased=True)
                else:
                    plt.errorbar(x_axis, perf, perf_error, marker='o', linestyle='--', label=strategy['display_name'])
            else:
                raise ValueError("Strategy {} not in cache, make sure collect_data() has ran first".format(strategy['display_name']))

        # plot setup
        plt.legend()
        plt.show()

    def plot_harmonic_mean_ranking_of_groups(self, kernel_indices=[0], y_metric='time', mean_absolute_error=True, combine_displayname_bargroup=False,
                                             displayname_as_group=True, remove_lcs=False, remove_substrings=['', '\'']):
        groups = defaultdict(list)
        groups_lower_err = defaultdict(list)
        groups_upper_err = defaultdict(list)
        groups_ranking = defaultdict(list)
        for kernel_index in kernel_indices:
            for strategy in self.strategies.values():
                if 'bar_group' not in strategy or strategy['bar_group'] == 'reference':
                    continue
                absolute_optimum = self.absolute_optima[kernel_index]
                # must use cached data written by collect_data() earlier
                expected_results = self.create_results_dict(strategy)
                cache = self.caches[kernel_index]
                cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                if cached_data is None:
                    raise ValueError("Strategy {} not in cache, make sure collect_data() has ran first".format(strategy['display_name']))
                results = cached_data['results']['results_per_number_of_evaluations']
                sum = 0
                perf = list()
                perf_err = list()
                for index, key in enumerate(results.keys()):
                    # skip the first 20 iterations because it is relatively noisy and dependent on initial sample
                    if index < 1:
                        continue
                    result = results[key]
                    diff = np.abs(result['mean_' + y_metric] - absolute_optimum)
                    if not mean_absolute_error:
                        diff = diff**2
                    sum += diff
                    perf.append(result['mean_' + y_metric])
                    perf_err.append(result['err_' + y_metric])
                n = len(results.keys()) - 1
                if mean_absolute_error:
                    rmse = sum / n
                    rmse_lower_err = rmse - (min(perf) - absolute_optimum)
                    rmse_upper_err = (max(perf) - absolute_optimum) - rmse
                else:
                    rmse = np.sqrt(sum / n)
                    rmse_lower_err = rmse - np.sqrt((absolute_optimum - min(perf))**2)
                    rmse_upper_err = np.sqrt((absolute_optimum - max(perf))**2) - rmse
                # add the value to the group
                group = f"{strategy['display_name']} {strategy['bar_group']}" if combine_displayname_bargroup else strategy[
                    'display_name'] if displayname_as_group else strategy['bar_group']
                # register in groups
                groups[group].append(rmse)
                groups_lower_err[group].append(rmse_lower_err)
                groups_upper_err[group].append(rmse_upper_err)
            # calculate the ranking
            rmses = list(groups[group] for group in groups.keys())
            # rmses_per_kernel = np.mean(rmses)
            rmses_per_kernel = list()
            for j in range(len(groups.keys())):
                rmses_per_kernel.append(np.mean(rmses[j]))
            # rank the indices
            temp = np.array(rmses_per_kernel).argsort()
            rankings_per_kernel = np.empty_like(temp)
            rankings_per_kernel[temp] = np.arange(len(rmses_per_kernel))
            for j, group in enumerate(groups.keys()):
                ranking = rankings_per_kernel[j] + 1
                groups_ranking[group].append(ranking)
                groups[group] = list()
        # compare the collected groups
        yval = list()
        yerr = list()
        labels = list()
        for group in groups.keys():
            ranking = groups_ranking[group]
            yval.append(statistics.harmonic_mean(ranking))
            yerr.append(np.std(ranking))
            labels.append(group)
            print(f"{group} | harmonic mean {round(statistics.harmonic_mean(ranking), 3)} | sd {round(np.std(ranking), 3)} ")
        # find the longest common substring among the labels to remove it
        for substring in remove_substrings:
            labels = list(label.replace(substring, '') for label in labels)
        lcs = self.longest_common_substring(labels)
        if remove_lcs and len(lcs) > 3:
            labels = list(label.replace(lcs, '') for label in labels)
        # plot
        # name = "Harmonic mean of rankings" if len(lcs) < 2 else lcs
        name = "harmonic_mean_ranking"
        fig = plt.figure(name)
        ax = fig.add_subplot(111)
        xaxis = range(len(yval))
        ax.bar(xaxis, yval, yerr=yerr, zorder=5)
        ax.set_xticks(xaxis)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Harmonic mean of rankings")
        ax.grid(axis='y', zorder=0, alpha=0.7)
        ax.set_ylim(0, ax.get_ylim()[1])
        fig.tight_layout()
        plt.show()

    def plot_mean_deviation_factor(self, kernel_indices=[0], y_metric='time', mean_absolute_error=True, combine_displayname_bargroup=False,
                                   displayname_as_group=True, remove_lcs=False, remove_substrings=['', '\'', 'Matern '], ignore_reference=False):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        strategies_ordered = self.order_strategies(main_bar_group='bo')
        groups = defaultdict(list)
        groups_lower_err = defaultdict(list)
        groups_upper_err = defaultdict(list)
        groups_factors = defaultdict(list)
        groups_factors_low_err = defaultdict(list)
        groups_factors_upp_err = defaultdict(list)
        for kernel_index in kernel_indices:
            for strategy in strategies_ordered:
                if 'bar_group' not in strategy:
                    continue
                if ignore_reference and strategy['bar_group'] == 'reference':
                    continue
                absolute_optimum = self.absolute_optima[kernel_index]
                # must use cached data written by collect_data() earlier
                expected_results = self.create_results_dict(strategy)
                cache = self.caches[kernel_index]
                cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                if cached_data is None:
                    raise ValueError("Strategy {} not in cache, make sure collect_data() has ran first".format(strategy['display_name']))
                results = cached_data['results']['results_per_number_of_evaluations']

                # Calculate the MAE / RMSE values and lower and upper errors
                counter = 0
                runs = results[list(results.keys())[0]][y_metric]
                RMSE_sums = [0] * len(runs)
                for index, key in enumerate(results.keys()):
                    # skip the first 20 iterations because it is relatively noisy and dependent on initial sample
                    if index < 1:
                        continue
                    counter += 1
                    result = results[key]
                    # calculate the seperate MAE sums
                    for index, res in enumerate(result[y_metric]):
                        diff = np.abs(res - absolute_optimum)
                        if not mean_absolute_error:
                            diff = diff**2
                        RMSE_sums[index] += diff
                RMSEs = list()
                for index, _ in enumerate(runs):
                    RMSEs.append(RMSE_sums[index] / counter)
                rmse = np.mean(RMSEs)
                rmse_lower_err, rmse_upper_err = calculate_lower_upper_error(RMSEs)

                # add the value to the group
                group = f"{strategy['display_name']} {strategy['bar_group']}" if combine_displayname_bargroup else strategy[
                    'display_name'] if displayname_as_group else strategy['bar_group']
                # register in groups
                groups[group].append(rmse)
                groups_lower_err[group].append(rmse_lower_err)
                groups_upper_err[group].append(rmse_upper_err)

            # calculate the ranking
            rmses = list(groups[group] for group in groups.keys())
            rmses_low_err = list(groups_lower_err[group] for group in groups.keys())
            rmses_upp_err = list(groups_upper_err[group] for group in groups.keys())
            rmses_per_kernel = list()
            rmses_low_err_per_kernel = list()
            rmses_upp_err_per_kernel = list()
            for j in range(len(groups.keys())):
                rmses_per_kernel.append(np.mean(rmses[j]))
                rmses_low_err_per_kernel.append(np.mean(rmses_low_err[j]))
                rmses_upp_err_per_kernel.append(np.mean(rmses_upp_err[j]))

            # calculate the factor from the average of the kernel
            average = np.mean(rmses_per_kernel)
            factor_from_average = list((rmse / average) for rmse in rmses_per_kernel)
            factor_from_average_low_err = list((rmse_low / average) for rmse_low in rmses_low_err_per_kernel)
            factor_from_average_upp_err = list((rmse_upp / average) for rmse_upp in rmses_upp_err_per_kernel)
            # print(
            #     f"Kernel {self.kernel_names[kernel_index]} (median {round(np.median(rmses_per_kernel), 3)}, mean {round(np.mean(rmses_per_kernel), 3)}): {list(round(rmse, 3) for rmse in rmses_per_kernel)} {list(round(factor, 3) for factor in factor_from_average)}"
            # )

            # write the factors to the groups
            for j, group in enumerate(groups.keys()):
                groups_factors[group].append(factor_from_average[j])
                groups_factors_low_err[group].append(factor_from_average_low_err[j])
                groups_factors_upp_err[group].append(factor_from_average_upp_err[j])
                groups[group] = list()
                groups_lower_err[group] = list()
                groups_upper_err[group] = list()

        # compare the collected groups
        yval = list()
        yerr = list()
        yerr_low = list()
        yerr_upp = list()
        labels = list()
        for group in groups.keys():
            factors = groups_factors[group]
            yval.append(np.mean(factors))
            yerr.append(np.std(factors))
            factors_low_err = groups_factors_low_err[group]
            factors_upp_err = groups_factors_upp_err[group]
            yerr_low.append(np.mean(factors_low_err))
            yerr_upp.append(np.mean(factors_upp_err))
            labels.append(group)
            print(f"{group} | mean deviation factor {round(np.mean(factors), 3)} | sd {round(np.std(factors), 3)} ")
        # find the longest common substring among the labels to remove it
        for substring in remove_substrings:
            labels = list(label.replace(substring, '') for label in labels)
        lcs = self.longest_common_substring(labels)
        if remove_lcs and len(lcs) > 3:
            labels = list(label.replace(lcs, '') for label in labels)
        # plot
        # name = "Harmonic mean of rankings" if len(lcs) < 2 else lcs
        name = "mean_deviation_factor"
        fig = plt.figure(name)
        ax = fig.add_subplot(111)
        xaxis = range(len(yval))
        ax.bar(xaxis, yval, yerr=yerr, zorder=5, color=colors[:len(self.strategies.values())])
        # ax.bar(xaxis, yval, yerr=(yerr_low, yerr_upp), zorder=5)
        ax.set_xticks(xaxis)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean deviation factor")
        ax.grid(axis='y', zorder=0, alpha=0.7)
        ax.set_ylim(0, ax.get_ylim()[1])
        fig.tight_layout()
        plt.show()

    def plot_mean_deviation_factor_heatmap(self, kernel_indices=[0], y_metric='time', mean_absolute_error=True, swap_axes=False, yaxislabel='Kernel',
                                           xaxislabel='Lengthscale', ignore_reference=False, remove_chars_from_labels=['\'']):
        """ Plot the mean deviation factors in a heatmap, with the display name and bar group along the axis """
        reserved_split_char = '%%'

        groups = defaultdict(list)
        groups_lower_err = defaultdict(list)
        groups_upper_err = defaultdict(list)
        groups_factors = defaultdict(list)
        groups_factors_low_err = defaultdict(list)
        groups_factors_upp_err = defaultdict(list)
        for kernel_index in kernel_indices:
            for strategy in self.strategies.values():
                if 'bar_group' not in strategy:
                    continue
                if ignore_reference and strategy['bar_group'] == 'reference':
                    continue
                absolute_optimum = self.absolute_optima[kernel_index]
                # must use cached data written by collect_data() earlier
                expected_results = self.create_results_dict(strategy)
                cache = self.caches[kernel_index]
                cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                if cached_data is None:
                    raise ValueError("Strategy {} not in cache, make sure collect_data() has ran first".format(strategy['display_name']))
                results = cached_data['results']['results_per_number_of_evaluations']

                # register the displayname and bargroup name

                # Calculate the MAE / RMSE values and lower and upper errors
                counter = 0
                runs = results[list(results.keys())[0]][y_metric]
                RMSE_sums = [0] * len(runs)
                for index, key in enumerate(results.keys()):
                    # skip the first 20 iterations because it is relatively noisy and dependent on initial sample
                    if index < 1:
                        continue
                    counter += 1
                    result = results[key]
                    # calculate the seperate MAE sums
                    for index, res in enumerate(result[y_metric]):
                        diff = np.abs(res - absolute_optimum)
                        if not mean_absolute_error:
                            diff = diff**2
                        RMSE_sums[index] += diff
                RMSEs = list()
                for index, _ in enumerate(runs):
                    RMSEs.append(RMSE_sums[index] / counter)
                rmse = np.mean(RMSEs)
                rmse_lower_err, rmse_upper_err = calculate_lower_upper_error(RMSEs)

                # add the value to the group
                group = f"{strategy['display_name']}{reserved_split_char}{strategy['bar_group']}"
                # register in groups
                groups[group].append(rmse)
                groups_lower_err[group].append(rmse_lower_err)
                groups_upper_err[group].append(rmse_upper_err)

            # calculate the ranking
            rmses = list(groups[group] for group in groups.keys())
            rmses_low_err = list(groups_lower_err[group] for group in groups.keys())
            rmses_upp_err = list(groups_upper_err[group] for group in groups.keys())
            rmses_per_kernel = list()
            rmses_low_err_per_kernel = list()
            rmses_upp_err_per_kernel = list()
            for j in range(len(groups.keys())):
                rmses_per_kernel.append(np.mean(rmses[j]))
                rmses_low_err_per_kernel.append(np.mean(rmses_low_err[j]))
                rmses_upp_err_per_kernel.append(np.mean(rmses_upp_err[j]))

            # calculate the factor from the average of the kernel
            average = np.mean(rmses_per_kernel)
            factor_from_average = list((rmse / average) for rmse in rmses_per_kernel)
            factor_from_average_low_err = list((rmse_low / average) for rmse_low in rmses_low_err_per_kernel)
            factor_from_average_upp_err = list((rmse_upp / average) for rmse_upp in rmses_upp_err_per_kernel)
            # print(
            #     f"Kernel {self.kernel_names[kernel_index]} (median {round(np.median(rmses_per_kernel), 3)}, mean {round(np.mean(rmses_per_kernel), 3)}): {list(round(rmse, 3) for rmse in rmses_per_kernel)} {list(round(factor, 3) for factor in factor_from_average)}"
            # )

            # write the factors to the groups
            for j, group in enumerate(groups.keys()):
                groups_factors[group].append(factor_from_average[j])
                groups_factors_low_err[group].append(factor_from_average_low_err[j])
                groups_factors_upp_err[group].append(factor_from_average_upp_err[j])
                groups[group] = list()
                groups_lower_err[group] = list()
                groups_upper_err[group] = list()

        # collect the displaynames and bargroups
        displaynames = list()
        bargroupnames = list()
        filter_chars = lambda char: "" if char in remove_chars_from_labels else char
        sanitize_label = lambda label: "".join(filter(filter_chars, label))
        for groupkey in groups_factors.keys():
            displayname, bargroupname = groupkey.split('%%')
            displayname = sanitize_label(displayname)
            bargroupname = sanitize_label(bargroupname)
            if displayname not in displaynames:
                displaynames.append(displayname)
            if bargroupname not in bargroupnames:
                bargroupnames.append(bargroupname)

        # split the combined groups into the displayname and bar group to build a 2D table
        dataframe = pd.DataFrame(index=displaynames, columns=bargroupnames)
        for groupkey in groups_factors.keys():
            displayname, bargroupname = groupkey.split('%%')
            displayname = sanitize_label(displayname)
            bargroupname = sanitize_label(bargroupname)
            value = np.mean(groups_factors[groupkey])
            dataframe[bargroupname] = dataframe[bargroupname].astype(float)
            dataframe.loc[displayname, bargroupname] = float(value)
        if swap_axes is True:
            dataframe = dataframe.transpose()

        # plot the data
        name = "mean_deviation_factor_heatmap"
        fig = plt.figure(name)
        ax = sns.heatmap(dataframe, annot=True, fmt='.3f', cmap='rocket_r', cbar_kws={ 'label': "Mean deviation factor"})
        if yaxislabel is not None and yaxislabel != '':
            ax.set_ylabel(yaxislabel)
        if xaxislabel is not None and xaxislabel != '':
            ax.set_xlabel(xaxislabel)
        fig.tight_layout()
        plt.show()

    # TODO add errorshade with plot_strategies_errorbar-calculation if possible and x_axis error if possible
    def surpasses(self, kernel_index=0, baselines=['bayes_opt_ei_CV_reference', 'bayes_opt_multi_reference', 'bayes_opt_multi-advanced_reference'],
                  surpassors=['extended_random_sample', 'extended_genetic_algorithm', 'extended_pso', 'extended_simulated_annealing',
                              'extended_firefly'], relative=True, errorbar=False, y_metric='time'):
        """ Check how much longer (if at all) it takes some methods to surpass some baseline methods """
        # plot setup
        figname = self.kernel_names[kernel_index]
        fig = plt.figure(figname)
        ax = fig.add_subplot(111)
        ax.grid()

        # setup data
        cache = self.caches[kernel_index]
        kernel_name = kernel_names[kernel_index]
        absolute_optimum = self.absolute_optima[kernel_index]
        relative_perf = lambda p: (p + 1) / (absolute_optimum + 1) if relative else p
        markers = ['P', 'd', 'v', 's', '*', 'X']
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # data collection
        for baseline_index, baseline in enumerate(baselines):
            marker = markers[baseline_index]
            baseline_strategy = self.strategies[baseline]
            baseline_strategy_str = baseline_strategy['display_name']
            baseline_expected_results = self.create_results_dict(baseline_strategy)
            baseline_cached_data = cache.get_strategy_results(baseline_strategy['name'], baseline_strategy['options'], baseline_strategy['repeats'],
                                                              baseline_expected_results)
            if baseline_cached_data is not None:
                baseline_results = baseline_cached_data['results']['results_per_number_of_evaluations']
                # baseline_max_key = str(max(list(int(k) for k in baseline_results.keys())))
                baseline_max_key = "220"
                baseline_max_result = baseline_results[baseline_max_key]
                baseline_max_perf = baseline_max_result['mean_' + y_metric]
                baseline_max_error = baseline_max_result['err_' + y_metric]
                baseline_label = f"{baseline_strategy_str}"
                y = relative_perf(baseline_max_perf)
                if errorbar:
                    y_err = baseline_max_perf(baseline_max_error)
                    ax.errorbar(int(baseline_max_key), y, yerr=y_err, label=baseline_label, fmt=marker, color='black')
                else:
                    ax.scatter(int(baseline_max_key), y, label=baseline_label, marker=marker, color='black')

                # iterate over the other methods
                for surpassor_index, surpassor in enumerate(surpassors):
                    color = colors[surpassor_index]
                    strategy = self.strategies[surpassor]
                    strategy_str = strategy['display_name']
                    # must use cached data written by collect_data() earlier
                    expected_results = self.create_results_dict(strategy)
                    cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                    if cached_data is not None:
                        results = cached_data['results']['results_per_number_of_evaluations']
                        label = f"{strategy_str}" if baseline_index == 0 else None
                        # work backwards from the end (1020) to the start (220), stopping when the performance is worse than the baseline
                        lastkey = str(max(list(int(k) for k in results.keys())))
                        lastperf = results[lastkey]['mean_' + y_metric]
                        lasterr = results[lastkey]['err_' + y_metric]
                        for key in reversed(results.keys()):
                            result = results[key]
                            perf = result['mean_' + y_metric]
                            err = result['err_' + y_metric]
                            y = relative_perf(lastperf)
                            y_err = relative_perf(lasterr)
                            # new error calculation
                            observations = list(relative_perf(res) for res in result[y_metric]) if relative else result[y_metric]
                            observations.sort()
                            middle_index = len(observations) // 2
                            middle_index_upper = middle_index + 1 if len(observations) % 2 != 0 else middle_index
                            lower_error = np.mean(observations[:middle_index])
                            upper_error = np.mean(observations[middle_index_upper:])
                            if int(key) < int(baseline_max_key):
                                print(
                                    f"Kernel {kernel_name}: Performance of {strategy_str} is already better than {baseline_strategy_str} at {baseline_max_key} iterations!"
                                )
                                if errorbar:
                                    ax.errorbar(int(baseline_max_key), y, yerr=y_err, label=label, fmt=marker, color=color)
                                else:
                                    # TODO ax.fill_between()
                                    ax.scatter(int(baseline_max_key), y, label=label, marker=marker, color=color)
                                break
                            if perf >= baseline_max_perf:
                                print(
                                    f"Kernel {kernel_name}: Performance of {strategy_str} is {round(perf, 3)} after {key} iterations. The performance of {baseline_strategy_str} is {round(baseline_max_perf, 3)} after {baseline_max_key} iterations."
                                )
                                if errorbar:
                                    ax.errorbar(int(lastkey), y, yerr=y_err, label=label, fmt=marker, color=color, ls='none')
                                else:
                                    # height = upper_error - lower_error
                                    # print(height)
                                    # print(err)
                                    # ellipse = Ellipse([int(lastkey), observations[middle_index]], height=height, width=20 * err, facecolor=color, alpha=0.1)
                                    # ax.add_artist(ellipse)
                                    # ax.fill_between([int(lastkey) - 10, int(lastkey) + 10], [lower_error, upper_error], color=color, alpha=0.1)
                                    ax.scatter(int(lastkey), y, label=label, marker=marker, color=color)
                                break
                            else:
                                lastkey = key
                                lastperf = perf
                                lasterr = err
                    else:
                        raise ValueError(f"{baseline_strategy_str} cache is invalid")
            else:
                raise ValueError(f"{strategy_str} cache is invalid")

        # finalize plot
        if absolute_optimum is not None:
            xlim = ax.get_xlim()
            ax.plot([xlim[0], xlim[1]], [1, 1], linestyle='-', label="True optimum {}".format(round(absolute_optimum, 3)))
            ax.set_xlim(xlim)
        ax.set_xlabel("Number of function evaluations used")
        if relative:
            ax.set_ylabel("Best found time relative to absolute optimum")
        else:
            ax.set_ylabel("Best found time in miliseconds")
        ax.legend()
        fig.tight_layout()
        plt.show()

    def surpasses_kernels(self, surpassors: list, kernel_indices=[0],
                          baselines=['bayes_opt_ei_CV_reference', 'bayes_opt_multi_reference',
                                     'bayes_opt_multi-advanced_reference'], relative=True, y_metric='time'):
        """ Check how much longer (if at all) it takes some methods to surpass some baseline methods """

        # data collection
        for baseline in baselines:
            baseline_strategy = self.strategies[baseline]
            baseline_strategy_str = baseline_strategy['display_name']

            # plot setup
            figname = f"Performance relative to {baseline_strategy_str}"
            fig = plt.figure(figname)
            ax = fig.add_subplot(111)
            ax.grid()

            for kernel_index in kernel_indices:
                kernel_name = kernel_names[kernel_index]
                absolute_optimum = self.absolute_optima[kernel_index]
                marker = self.kernel_defaults[kernel_name]['marker']

                cache = self.caches[kernel_index]
                baseline_expected_results = self.create_results_dict(baseline_strategy)
                baseline_cached_data = cache.get_strategy_results(baseline_strategy['name'], baseline_strategy['options'], baseline_strategy['repeats'],
                                                                  baseline_expected_results)
                if baseline_cached_data is not None:
                    baseline_results = baseline_cached_data['results']['results_per_number_of_evaluations']
                    baseline_max_key = str(max(list(int(k) for k in baseline_results.keys())))
                    baseline_max_result = baseline_results[baseline_max_key]
                    baseline_max_perf = round(baseline_max_result['mean_' + y_metric], 3)
                    baseline_label = f"{kernel_name} {baseline_strategy_str}" if len(kernel_indices) > 1 else f"{baseline_strategy_str}"
                    y = (baseline_max_perf + 1) / (absolute_optimum + 1) if relative else baseline_max_perf
                    ax.scatter(int(baseline_max_key), y, label=baseline_label, marker=marker)
                    for surpassor in surpassors:
                        strategy = self.strategies[surpassor]
                        strategy_str = strategy['display_name']
                        # must use cached data written by collect_data() earlier
                        expected_results = self.create_results_dict(strategy)
                        cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                        if cached_data is not None:
                            results = cached_data['results']['results_per_number_of_evaluations']
                            label = f"{kernel_name} {strategy_str}" if len(kernel_indices) > 1 else f"{strategy_str}"
                            for key in reversed(results.keys()):
                                result = results[key]
                                perf = round(result['mean_' + y_metric], 3)
                                y = (perf + 1) / (absolute_optimum + 1) if relative else perf
                                if int(key) < int(baseline_max_key):
                                    print(
                                        f"Kernel {kernel_name}: Performance of {strategy_str} is already better than {baseline_strategy_str} at {baseline_max_key} iterations!"
                                    )
                                    ax.scatter(int(baseline_max_key), y, label=label, marker=marker)
                                    break
                                if perf > baseline_max_perf:
                                    print(
                                        f"Kernel {kernel_name}: Performance of {strategy_str} is {perf} after {key} iterations. The performance of {baseline_strategy_str} is {baseline_max_perf} after {baseline_max_key} iterations."
                                    )
                                    ax.scatter(int(key), y, label=label, marker=marker)
                                    break
                        else:
                            raise ValueError(f"{baseline_strategy_str} cache is invalid")
                else:
                    raise ValueError(f"{strategy_str} cache is invalid")

            # finalize plot
            ax.set_xlabel("Number of iterations")
            if relative:
                ax.set_ylabel("Best found time relative to absolute optimum")
            else:
                ax.set_ylabel("Best found time in miliseconds")
            ax.legend()
            fig.tight_layout()
            plt.show()

    def longest_common_substring(self, names: list) -> str:
        """ from https://stackoverflow.com/questions/58585052/find-most-common-substring-in-a-list-of-strings """
        import operator
        from difflib import SequenceMatcher
        substring_counts = {}
        for i in range(0, len(names)):
            for j in range(i + 1, len(names)):
                string1 = names[i]
                string2 = names[j]
                match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
                matching_substring = string1[match.a:match.a + match.size]
                if (matching_substring not in substring_counts):
                    substring_counts[matching_substring] = 1
                else:
                    substring_counts[matching_substring] += 1
        if len(substring_counts.items()) == 0:
            return ""
        longest_common_substrings = max(substring_counts.items(), key=operator.itemgetter(1))
        return longest_common_substrings[0] if len(longest_common_substrings) > 0 else ''


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("-kernels", nargs="*", type=str, help="List of kernel names to experiment")
    CLI.add_argument("-devices", nargs="*", type=str, default=["GTX_TITAN_X"], help="List of devices to experiment")
    args = CLI.parse_args()
    kernel_names = args.kernels
    device_names = args.devices

    if kernel_names is None:
        raise ValueError("Invalid '-kernels' option. Run 'experiments.py -h' to read more about the options.")
    if device_names is None:
        raise ValueError("Invalid '-devices' option. Run 'experiments.py -h' to read more about the options.")

    devices = {
        'A100': {
            'name': 'A100',
            'displayname': 'A100',
            'absolute_optima': [8.518111999999999, 0.7390080038458109, 13.090592086315155],
            'y_axis_upper_limit': [9.9, 1.0, 13.45],
        },
        'RTX_2070_Super': {
            'name': 'RTX_2070_Super',
            'displayname': 'RTX 2070 Super',
            'absolute_optima': [17.111732999999997, 1.2208920046687126, 12.325379967689514],
            'y_axis_upper_limit': [22.8, 1.9, 13.5],
        },
        'GTX_TITAN_X': {
            'name': 'GTX_TITAN_X',
            'displayname': 'GTX Titan X',
            'absolute_optima': [28.307017000000002, 1.6253190003335476, 26.968406021595],
            'y_axis_upper_limit': [37, 2.52, 35.1],
        },
    }

    for device_name in device_names:
        device = devices[device_name]
        if len(device_names) > 1:
            print(f"\n\n\n      |---| {device['displayname']} |---| \n")
        defaults = {
            'GEMM': {
                'dev_name': device['name'],
                'absolute_optimum': device['absolute_optima'][0],
                'y_axis_upper_limit': device['y_axis_upper_limit'][0],
                'marker': 'v',
            },
            'convolution': {
                'dev_name': device['name'],
                'absolute_optimum': device['absolute_optima'][1],
                'y_axis_upper_limit': device['y_axis_upper_limit'][1],
                'marker': 's',
            },
            'pnpoly': {
                'dev_name': device['name'],
                'absolute_optimum': device['absolute_optima'][2],
                'y_axis_upper_limit': device['y_axis_upper_limit'][2],
                'marker': '*',
            },
            'expdist': {
                'dev_name': 'A100',
                'absolute_optimum': 33.87768375922341,
                'y_axis_upper_limit': 47,
                'marker': '1',
            },
            'adding': {
                'cache_name': 'sw_adding_kernel<double, 0>',
                'dev_name': 'A100',
                'absolute_optimum': 1.4682930000126362,
                'y_axis_upper_limit': None,
                'marker': '2',
            },
            'Rosenbrock': {
                'dev_name': 'generator',
                'absolute_optimum': 0.0,
                'y_axis_upper_limit': 4,
                'marker': 'd',
            },
            'Mishrasbird': {
                'dev_name': 'generator',
                'absolute_optimum': 0.0,
                'y_axis_upper_limit': 51,
                'marker': 'P',
            },
            'Gomez-Levy': {
                'dev_name': 'generator',
                'absolute_optimum': 0.0,
                'y_axis_upper_limit': 0.27,
                'marker': 'X',
            },
        }

        device_names = list()
        absolute_optima = list()
        for kernel_name in kernel_names:
            if kernel_name not in defaults:
                raise ValueError(f"Kernel {kernel_name} not in defaults dict")
            default = defaults[kernel_name]
            device_names.append(default['dev_name'])
            absolute_optima.append(default['absolute_optimum'])

        try:
            # Collect data
            # change_dir("../kernel_tuner_simulation")
            change_dir("../cached_data_used")
            kernels = list()
            for kernel_name in kernel_names:
                kernels.append(importlib.import_module(kernel_name))
            stats = StatisticalData(kernels, kernel_names, device_names, absolute_optima, kernel_defaults=defaults)

            # Plot
            # stats.cache.delete()
            # stats.plot_RMSE_barchart(range(len(kernel_names)))
            # stats.surpasses_kernels(range(len(kernel_names)))
            for kernel_index in range(len(kernel_names)):
                pass
                # print(f"Plotting {kernel_names[kernel_index]}")
                stats.plot_strategies_errorbar(kernel_index=kernel_index, x_metric='num_evals', y_metric='time', plot_errors=False)
                # stats.plot_RMSE_barchart(kernel_indices=[kernel_index], log_scale=False)
                # stats.surpasses(
                #     kernel_index=kernel_index, surpassors=[
                #         'extended_random_sample_rem_unique', 'extended_simulated_annealing_rem_unique', 'extended_mls_rem_unique',
                #         'extended_genetic_algorithm_rem_unique'
                #     ])
                # stats.surpasses(
                #     baselines=['bayes_opt_multi-advanced_reference'], kernel_index=kernel_index, surpassors=[
                #         'extended_random_sample_rem_unique', 'extended_simulated_annealing_rem_unique', 'extended_mls_rem_unique',
                #         'extended_genetic_algorithm_rem_unique'
                #     ])
            # stats.plot_harmonic_mean_ranking_of_groups(kernel_indices=range(len(kernel_names)))
            stats.plot_mean_deviation_factor(kernel_indices=range(len(kernel_names)))
            # stats.plot_mean_deviation_factor_heatmap(kernel_indices=range(len(kernel_names)))
            # stats.plot_mean_deviation_factor_heatmap(kernel_indices=range(len(kernel_names)), swap_axes=False, xaxislabel='', yaxislabel='')
            # stats.cache.delete()
        except KeyboardInterrupt:
            print("Keyboard interrupt, exiting")
            exit(0)
