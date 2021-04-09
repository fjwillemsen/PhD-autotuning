from caching import CacheInterface, CachedObject
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import scipy
import progressbar
import time as python_time

import importlib
import os
import sys


def change_dir(dir: str):
    os.chdir(dir)
    sys.path.append(dir)


class StatisticalData():
    """ Object that captures all statistical data and functions to visualize, plots have possible metrics 'GFLOP/s' or 'time' """

    x_metric_displayname = dict({
        'num_evals': 'Number of evaluations used',
        'strategy_time': 'Average time taken by strategy in miliseconds',
        'compile_time': 'Average compile time in miliseconds',
        'execution_time': 'Evaluation execution time taken in miliseconds',
        'total_time': 'Average total time taken in miliseconds',
    })

    y_metric_displayname = dict({
        'time': 'Best found time in miliseconds',
        'GFLOP/s': 'GFLOP/s',
    })

    def __init__(self, kernel, kernel_name, device_name, simulation_mode=True):
        self.kernel = kernel
        self.kernel_name = kernel_name
        self.device_name = device_name
        self.simulation_mode = simulation_mode

        # setup the strategies (beware that the options determine the maximum number of iterations, so setting this lower than the num_of_evaluations causes problems)
        default_number_of_repeats = 8
        # default_number_of_evaluations = np.array([25, 50, 75, 100, 125, 150, 175, 200]).astype(int)
        default_number_of_evaluations = np.array([5, 10, 15, 20, 25, 50, 75, 100]).astype(int)
        self.strategies = {
        # 'brute_force': {
        #     'name': 'brute_force',
        #     'display_name': 'Brute Force',
        #     'nums_of_evaluations': np.array([]).astype(int),
        #     'repeats': 1,
        #     'options': {},
        # },
            'random_sample': {
                'name': 'random_sample',
                'display_name': 'Random Sample',
                'nums_of_evaluations': default_number_of_evaluations,
                'repeats': 20,
                'options': {
                    'fraction': 0.1
                }
            },
            'genetic_algorithm': {
                'name': 'genetic_algorithm',
                'display_name': 'Genetic Algorithm',
                'nums_of_evaluations': default_number_of_evaluations,
                'repeats': default_number_of_repeats,
                'options': {
                    'maxiter': max(default_number_of_evaluations),
                }
            },
            'firefly': {
                'name': 'firefly_algorithm',
                'display_name': 'Firefly Algorithm',
                'nums_of_evaluations': default_number_of_evaluations,
                'repeats': default_number_of_repeats,
                'options': {},
            },
            'pso': {
                'name': 'pso',
                'display_name': 'Particle Swarm Optimization',
                'nums_of_evaluations': default_number_of_evaluations,
                'repeats': default_number_of_repeats,
                'options': {},
            },
        # 'bayes_opt': {
        #     'name': 'bayes_opt',
        #     'display_name': 'Bayesian Optimization',
        #     'nums_of_evaluations': default_number_of_evaluations,
        #     'repeats': 3,
        #     'options': {
        #         'maxiter': max(default_number_of_evaluations),
        #     },
        # },
        # 'bayes_opt_old': {
        #     'name': 'bayes_opt_old',
        #     'display_name': 'Bayesian Optimization (old)',
        #     'nums_of_evaluations': default_number_of_evaluations,
        #     'repeats': 3,
        #     'options': {
        #         'maxiter': max(default_number_of_evaluations),
        #     },
        # }
        }

        self.collect_data()

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

    def collect_data(self):
        """ Executes strategies to obtain (or retrieve from cache) the statistical data """
        # get or create a cache
        self.cache = CachedObject(self.kernel_name, self.device_name, deepcopy(self.strategies))

        # run all strategies
        for strategy in self.strategies.values():
            print("Running {}".format(strategy['display_name']))

            # if the strategy is in the cache, use cached data
            expected_results = self.create_results_dict(strategy)
            cached_data = self.cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
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
                res, _ = self.kernel.tune(device_name=self.device_name, strategy=strategy['name'], strategy_options=strategy['options'], verbose=False,
                                          quiet=True, simulation_mode=self.simulation_mode)
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
            print("Total time: {} ms, std {}".format(round(total_time_mean, 3), round(total_time_std, 3)))

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
                result['mean_cumulative_strategy_time'] = 0
                result['mean_cumulative_compile_time'] = 0
                result['mean_cumulative_execution_time'] = np.mean(result['cumulative_execution_time'])
                cumulative_total_time = total_time_mean_per_eval * num_of_evaluations
                if self.simulation_mode:
                    result['mean_cumulative_total_time'] = cumulative_total_time + result['mean_cumulative_strategy_time'] + result[
                        'mean_cumulative_compile_time'] + result['mean_cumulative_execution_time']
                else:
                    result['mean_cumulative_total_time'] = cumulative_total_time
                # if result['err_actual_num_evals'] > 0:
                #     raise ValueError('The number of actual evaluations over the runs has varied: {}'.format(result['actual_num_evals']))
                # if result['mean_actual_num_evals'] != num_of_evaluations:
                #     print(
                #         "The set number of evaluations ({}) is not equal to the actual number of evaluations ({}). Try increasing the fraction or maxiter in strategy options."
                #         .format(num_of_evaluations, result['mean_actual_num_evals']))

            # write to the cache
            self.cache.set_strategy(deepcopy(strategy), results_to_write)
            # print("")

    def plot_strategies_errorbar(self, x_metric='num_evals', y_metric='GFLOP/s', shaded=True):
        """ Plots all strategies with errorbars, shaded plots a shaded error region instead of error bars. Y-axis and X-axis metrics can be chosen. """
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
                    # calculate y axis
                    perf = np.append(perf, result['mean_' + y_metric])
                    perf_error = np.append(perf_error, result['err_' + y_metric])
                    # calculate x axis
                    actual_num_evals = np.append(actual_num_evals, result['mean_actual_num_evals'])
                    cumulative_strategy_time = np.append(cumulative_strategy_time, result['mean_cumulative_strategy_time'])
                    cumulative_compile_time = np.append(cumulative_compile_time, result['mean_cumulative_compile_time'])
                    cumulative_execution_time = np.append(cumulative_execution_time, result['mean_cumulative_execution_time'])
                    cumulative_total_time = np.append(cumulative_total_time, result['mean_cumulative_total_time'])
                # set x axis
                x_axis = actual_num_evals
                if x_metric == 'strategy_time':
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
        plt.xlabel(self.x_metric_displayname[x_metric])
        plt.ylabel(self.y_metric_displayname[y_metric])
        plt.legend()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 1 and len(sys.argv) != 3:
        print("Usage: visualize.py [kernel name] [device name]")
        exit(1)

    if len(sys.argv) > 1:
        kernel_name = sys.argv[1] or None
        device_name = sys.argv[2] or None
        # change_dir("../cached_runs")
        change_dir("../kernel_tuner_simulation")
        kernel = importlib.import_module(kernel_name)
        stats = StatisticalData(kernel, kernel_name, device_name=device_name)
        # stats.cache.delete()
        stats.plot_strategies_errorbar(x_metric='execution_time', y_metric='time', shaded=True)
        # stats.cache.delete()
    else:
        raise ValueError("Bad arguments, expected: visualize.py [kernel name] [device name]")
