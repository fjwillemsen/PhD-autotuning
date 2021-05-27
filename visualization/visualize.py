from caching import CacheInterface, CachedObject
from collections import OrderedDict, defaultdict
from copy import deepcopy
import matplotlib.pyplot as plt
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
    variation_types = ['acquisition_functions_exploration', 'covariance_function']
    if variation_type not in variation_types:
        raise ValueError(f"{variation_type} not in {variation_types}")
    acquisition_functions = ['ei', 'poi', 'lcb']
    acquisition_functions_display_names = ['EI', 'PoI', 'LCB', 'LCB-S']
    variations = [0.01, 0.1, 0.3, 'CV']
    group_by_variation = True
    variation_in_display_name = False
    reference_strategies = {
        'random_sample': {
            'name': 'random_sample',
            'strategy': 'random_sample',
            'display_name': 'Random Sample',
            'bar_group': 'reference',
            'nums_of_evaluations': default_number_of_evaluations,
            'repeats': 50,
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
            'repeats': default_number_of_repeats,
            'options': {
                'max_fevals': max_num_evals,
            }
        },
    }
    strategies = reference_strategies
    for af_index, af in enumerate(acquisition_functions):
        for variation in variations:
            key = f"{strategy_name}_{af}_{variation}"
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
            if variation_type == 'covariance_function':
                options['covariancekernel'] = variation
            dct['options'] = options
            # write to strategies
            strategies[key] = dct
    return strategies


class StatisticalData():
    """ Object that captures all statistical data and functions to visualize, plots have possible metrics 'GFLOP/s' or 'time' """

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

    def __init__(self, kernels: list, kernel_names: list, device_names: list, absolute_optima: list, simulation_mode=True):
        self.kernel = kernels
        self.kernel_names = kernel_names
        self.absolute_optima = absolute_optima
        self.device_names = device_names
        self.simulation_mode = simulation_mode

        # print(generate_strategies(variation_type='acquisition_functions_exploration'))
        self.strategies = {
            'random_sample': {
                'name': 'random_sample',
                'strategy': 'random_sample',
                'display_name': 'Random Sample',
                'bar_group': 'reference',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 50,
                'options': {
                    'fraction': 0.05
                }
            },
            'genetic_algorithm': {
                'name': 'genetic_algorithm',
                'strategy': 'genetic_algorithm',
                'display_name': 'Genetic Algorithm',
                'bar_group': 'reference',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220
                }
            },
            'bayes_opt_ei_0.01': {
                'name': 'bayes_opt_ei_0.01',
                'strategy': 'bayes_opt',
                'display_name': 'EI',
                'bar_group': '0.01',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'ei',
                    'methodparams': {
                        'explorationfactor': 0.01,
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_ei_0.1': {
                'name': 'bayes_opt_ei_0.1',
                'strategy': 'bayes_opt',
                'display_name': 'EI',
                'bar_group': '0.1',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'ei',
                    'methodparams': {
                        'explorationfactor': 0.1,
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_ei_0.3': {
                'name': 'bayes_opt_ei_0.3',
                'strategy': 'bayes_opt',
                'display_name': 'EI',
                'bar_group': '0.3',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'ei',
                    'methodparams': {
                        'explorationfactor': 0.3,
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_ei_CV': {
                'name': 'bayes_opt_ei_CV',
                'strategy': 'bayes_opt',
                'display_name': 'EI',
                'bar_group': 'CV',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'ei',
                    'methodparams': {
                        'explorationfactor': 'CV',
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_poi_0.01': {
                'name': 'bayes_opt_poi_0.01',
                'strategy': 'bayes_opt',
                'display_name': 'PoI',
                'bar_group': '0.01',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'poi',
                    'methodparams': {
                        'explorationfactor': 0.01,
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_poi_0.1': {
                'name': 'bayes_opt_poi_0.1',
                'strategy': 'bayes_opt',
                'display_name': 'PoI',
                'bar_group': '0.1',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'poi',
                    'methodparams': {
                        'explorationfactor': 0.1,
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_poi_0.3': {
                'name': 'bayes_opt_poi_0.3',
                'strategy': 'bayes_opt',
                'display_name': 'PoI',
                'bar_group': '0.3',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'poi',
                    'methodparams': {
                        'explorationfactor': 0.3,
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_poi_CV': {
                'name': 'bayes_opt_poi_CV',
                'strategy': 'bayes_opt',
                'display_name': 'PoI',
                'bar_group': 'CV',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'poi',
                    'methodparams': {
                        'explorationfactor': 'CV',
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_lcb_0.01': {
                'name': 'bayes_opt_lcb_0.01',
                'strategy': 'bayes_opt',
                'display_name': 'LCB',
                'bar_group': '0.01',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'lcb',
                    'methodparams': {
                        'explorationfactor': 0.01,
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_lcb_0.1': {
                'name': 'bayes_opt_lcb_0.1',
                'strategy': 'bayes_opt',
                'display_name': 'LCB',
                'bar_group': '0.1',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'lcb',
                    'methodparams': {
                        'explorationfactor': 0.1,
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_lcb_0.3': {
                'name': 'bayes_opt_lcb_0.3',
                'strategy': 'bayes_opt',
                'display_name': 'LCB',
                'bar_group': '0.3',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'lcb',
                    'methodparams': {
                        'explorationfactor': 0.3,
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_lcb_CV': {
                'name': 'bayes_opt_lcb_CV',
                'strategy': 'bayes_opt',
                'display_name': 'LCB',
                'bar_group': 'CV',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'lcb',
                    'methodparams': {
                        'explorationfactor': 'CV',
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            },
            'bayes_opt_lcb-srinivas': {
                'name': 'bayes_opt_lcb-srinivas',
                'strategy': 'bayes_opt',
                'display_name': 'LCB Srinivas',
                'bar_group': 'LCB Srinivas',
                'nums_of_evaluations': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220],
                'repeats': 7,
                'options': {
                    'max_fevals': 220,
                    'method': 'lcb-srinivas',
                    'methodparams': {
                        'explorationfactor': 0.1,
                        'zeta': 1,
                        'skip_duplicate_after': 30
                    }
                }
            }
        }

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

    def collect_data(self, kernel, kernel_name, device_name) -> CachedObject:
        """ Executes strategies to obtain (or retrieve from cache) the statistical data """
        # get or create a cache
        cache = CachedObject(kernel_name, device_name, deepcopy(self.strategies))

        # run all strategies
        for strategy in self.strategies.values():
            print(f"Running {strategy['display_name']} {strategy['bar_group'] if 'bar_group' in strategy else ''}")

            # if the strategy is in the cache, use cached data
            if 'ignore_cache' not in strategy:
                expected_results = self.create_results_dict(strategy)
                cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                if cached_data is not None:
                    print("| retrieved from cache")
                    continue

            # repeat the strategy as specified
            repeated_results = list()
            total_time_results = np.array([])
            nums_of_evaluations = strategy['nums_of_evaluations']
            for rep in progressbar.progressbar(range(strategy['repeats']), redirect_stdout=True):
                # print(rep + 1, end=', ', flush=True)
                total_start_time = python_time.perf_counter()
                warnings.simplefilter("ignore", UserWarning)
                res, _ = kernel.tune(device_name=device_name, strategy=strategy['strategy'], strategy_options=strategy['options'], verbose=False, quiet=True,
                                     simulation_mode=self.simulation_mode)
                warnings.simplefilter("default", UserWarning)
                total_end_time = python_time.perf_counter()
                repeated_results.append(res)
                total_time_ms = round((total_end_time - total_start_time) * 1000)
                total_time_results = np.append(total_time_results, total_time_ms)
                if len(strategy['nums_of_evaluations']) <= 0:
                    nums_of_evaluations = np.append(nums_of_evaluations, len(res))

            # create the results dict for this strategy
            strategy['nums_of_evaluations'] = nums_of_evaluations
            results_to_write = self.create_results_dict(strategy)
            total_time_mean = np.mean(total_time_results)
            total_time_std = np.std(total_time_results)
            total_time_mean_per_eval = total_time_mean / nums_of_evaluations[-1]
            print("Total mean time: {} ms, std {}".format(round(total_time_mean, 3), round(total_time_std, 3)))

            # for every number of evaluations specified, find the best and collect details on it
            for res in repeated_results:
                for num_of_evaluations in nums_of_evaluations:
                    limited_res = res[:num_of_evaluations]
                    best = min(limited_res, key=lambda x: x['time'])
                    time = best['time']
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

    # def calc_nrows_ncols(self, n):
    #     if n == 1:
    #         return (1, 1)
    #     nrows = math.sqrt(n)

    def plot_strategies_errorbar(self, kernel_index=0, x_metric='num_evals', y_metric='GFLOP/s', shaded=True):
        """ Plots all strategies with errorbars, shaded plots a shaded error region instead of error bars. Y-axis and X-axis metrics can be chosen. """
        # if len(self.caches) != 1:
        #     raise ValueError("This function does not support plotting multiple kernels.")
        print(f"Plotting {self.kernel_names[kernel_index]}")
        cache = self.caches[kernel_index]
        absolute_optimum = self.absolute_optima[kernel_index]
        if absolute_optimum is not None:
            plt.plot([self.min_num_evals, self.max_num_evals], [absolute_optimum, absolute_optimum], linestyle='-',
                     label="True optimum {}".format(round(absolute_optimum, 3)))

        for strategy in self.strategies.values():
            collect_xticks = list()
            # must use cached data written by collect_data() earlier
            expected_results = self.create_results_dict(strategy)
            cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
            if cached_data is not None:
                results = cached_data['results']['results_per_number_of_evaluations']
                perf = np.array([])
                perf_error = np.array([])
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
                    plt.fill_between(x_axis, perf - perf_error, perf + perf_error, alpha=0.2, antialiased=True)
                    plt.plot(x_axis, perf, marker='o', linestyle='--', label=strategy['display_name'])
                else:
                    plt.errorbar(x_axis, perf, perf_error, marker='o', linestyle='--', label=strategy['display_name'])
            else:
                raise ValueError("Strategy {} not in cache, make sure collect_data() has ran first".format(strategy['display_name']))

        # plot setup
        # plt.ylim(0, 2)
        if len(collect_xticks) > 0:
            plt.xticks(collect_xticks[0])
        plt.xlabel(self.x_metric_displayname[x_metric])
        plt.ylabel(self.y_metric_displayname[y_metric])
        plt.legend()
        plt.show()

    def plot_RMSE_barchart(self, kernel_indices=[0], y_metric='time', error_bars=True, bar_width=0.1, ref_value_in_legend=False, log=False):
        """ Root Mean Squared Error barchart, skipping the first set of evaluations to avoid weirdness. Can compare multiple kernels if they have the same range of results. """
        if log:
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
                sum = 0
                perf = list()
                perf_err = list()
                for index, key in enumerate(results.keys()):
                    # skip the first 20 iterations because it is relatively noisy and dependent on initial sample
                    if index < 1:
                        continue
                    result = results[key]
                    diff = ((absolute_optimum - result['mean_' + y_metric])**2)
                    sum += diff
                    perf.append(result['mean_' + y_metric])
                    perf_err.append(result['err_' + y_metric])
                n = len(results.keys()) - 1
                rmse = np.sqrt(sum / n)
                # rmse = np.mean(perf)
                rmse_lower_err = rmse - np.sqrt((absolute_optimum - min(perf))**2)
                rmse_upper_err = np.sqrt((absolute_optimum - max(perf))**2) - rmse
                if log:
                    print(f"{strategy['display_name']}: {rmse}")
                # add the value to the group
                if len(kernel_indices) == 1:
                    group = strategy['bar_group'] if 'bar_group' in strategy else 'default'
                else:
                    group = strategy['display_name']
                groups[group].append(rmse)
                groups_lower_err[group].append(rmse_lower_err)
                groups_upper_err[group].append(rmse_upper_err)
                # add the name to the xtick list
                if len(kernel_indices) == 1:
                    name = strategy['display_name']
                else:
                    name = self.kernel_names[kernel_index]
                if group == 'reference':
                    ref_names.append(name)
                else:
                    if name not in names:
                        names.append(name)
        # plotting
        index = np.array([0])
        offset = -bar_width
        # plot the barchart groups
        for group in groups.keys():
            if group == 'reference':
                continue
            offset += bar_width
            index = np.arange(len(groups[group])) + offset
            label = group if group != 'default' else ''
            if error_bars is True:
                plt.bar(index, groups[group], yerr=(groups_lower_err[group], groups_upper_err[group]), width=bar_width, zorder=5, label=label, alpha=0.8,
                        error_kw=dict(lw=1, capsize=2))
            else:
                plt.bar(index, groups[group], width=bar_width, zorder=5, label=label, alpha=0.8)
        # plot the reference lines
        if 'reference' in groups:
            for i, ref in enumerate(groups['reference']):
                offset_ref = bar_width if len(index) <= 1 else offset
                label = f"{ref_names[i]} ({round(ref, 3)})" if ref_value_in_legend else ref_names[i]
                plt.plot([min(index) - offset_ref, max(index) + offset_ref], [ref, ref], linestyle='--', label=label)
        plt.ylabel('RMSE')
        # plt.yscale('log')
        plt.xticks(index - offset / 2, names)
        plt.grid(axis='y', zorder=0, alpha=0.7)
        # plt.legend(loc='upper left')
        plt.legend()
        plt.tight_layout()
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


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    # CLI.add_argument("--device_name", help="Device name to run on", default=None)
    # CLI.add_argument("--absolute_optima", nargs="*", type=float, default=[], help="List of absolute optima to be used per kernel")
    # device_name = args.device_name
    # absolute_optima = args.absolute_optima
    CLI.add_argument("-kernels", nargs="*", type=str, help="List of kernel names to be visualized")
    args = CLI.parse_args()
    kernel_names = args.kernels

    default_dev_name = 'GTX_TITAN_X'
    # dev_absolute_optima = [17.111733, 1.220892, 12.32538]    # RTX_2070_Super
    dev_absolute_optima = [28.307017000000002, 1.6253190003335476, 26.968406021595]    # GTX_TITAN_X
    defaults = {
        'GEMM': {
            'dev_name': default_dev_name,
            'absolute_optimum': dev_absolute_optima[0],
        },
        'convolution': {
            'dev_name': default_dev_name,
            'absolute_optimum': dev_absolute_optima[1],
        },
        'pnpoly': {
            'dev_name': default_dev_name,
            'absolute_optimum': dev_absolute_optima[2],
        },
        'Rosenbrock': {
            'dev_name': 'generator',
            'absolute_optimum': 0.0,
        },
        'Mishrasbird': {
            'dev_name': 'generator',
            'absolute_optimum': 0.0,
        },
        'Gomez-Levy': {
            'dev_name': 'generator',
            'absolute_optimum': 0.0,
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

    # Collect data
    # change_dir("../cached_runs")
    change_dir("../kernel_tuner_simulation")
    kernels = list()
    for kernel_name in kernel_names:
        kernels.append(importlib.import_module(kernel_name))
    stats = StatisticalData(kernels, kernel_names, device_names, absolute_optima)

    # Plot
    # stats.cache.delete()
    # stats.plot_RMSE_barchart(range(len(kernel_names)))
    for kernel_index in range(len(kernel_names)):
        print(f"Plotting {kernel_names[kernel_index]}")
        stats.plot_RMSE_barchart(kernel_indices=[kernel_index])
        # stats.plot_strategies_errorbar(kernel_index, x_metric='num_evals', y_metric='time', shaded=True)
    # stats.cache.delete()
