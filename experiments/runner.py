""" Interface to run an experiment on Kernel Tuner """
from copy import deepcopy
import numpy as np
import progressbar
from typing import Any, Tuple
import time as python_time
import warnings
import yappi

from metrics import units, quantity
from caching import CachedObject

record_data = ['mean_actual_num_evals']


def remove_duplicates(res: list, remove_duplicate_results: bool):
    """ Removes duplicate configurations from the results """
    if not remove_duplicate_results:
        return res
    unique_res = list()
    for result in res:
        if result not in unique_res:
            unique_res.append(result)
    return unique_res


def tune(kernel, kernel_name: str, device_name: str, strategy: dict, tune_options: dict, profiling: bool) -> Tuple[list, int]:
    """ Execute a strategy, return the result, runtime and optional profiling statistics """

    def tune_with_kerneltuner():
        """ interface with kernel tuner to tune the kernel and return the results """
        if profiling:
            yappi.set_clock_type("cpu")
            yappi.start()
        res, env = kernel.tune(device_name=device_name, strategy=strategy['strategy'], strategy_options=strategy['options'], **tune_options)
        if profiling:
            yappi.stop()
        return res, env

    total_start_time = python_time.perf_counter()
    warnings.simplefilter("ignore", UserWarning)
    try:
        res, _ = tune_with_kerneltuner()
    except ValueError:
        print("Something went wrong, trying once more.")
        res, _ = tune_with_kerneltuner()
    warnings.simplefilter("default", UserWarning)
    total_end_time = python_time.perf_counter()
    total_time_ms = round((total_end_time - total_start_time) * 1000)
    # TODO when profiling, should the total_time_ms not be the time from profiling_stats? Otherwise we're timing the profiling code as well
    return res, total_time_ms


def collect_results(kernel, kernel_name: str, device_name: str, strategy: dict, expected_results: dict, default_records: dict, profiling: bool,
                    optimization='time', remove_duplicate_results=True) -> dict:
    """ Executes strategies to obtain (or retrieve from cache) the statistical data """
    print(f"Running {strategy['display_name']}")
    nums_of_evaluations = strategy['nums_of_evaluations']
    max_num_evals = min(strategy['nums_of_evaluations'])
    # TODO put the tune options in the .json in strategy_defaults?
    tune_options = {
        'verbose': False,
        'quiet': True,
        'simulation_mode': True
    }

    def report_multiple_attempts(rep: int, len_res: int, len_unique_res: int, strategy_repeats: int):
        """ If multiple attempts are necessary, report the reason """
        if len_res < 1:
            print(f"({rep+1}/{strategy_repeats}) No results found, trying once more...")
        elif len_unique_res < max_num_evals:
            print(f"Too few unique results found ({len_unique_res} in {len_res} evaluations), trying once more...")
        else:
            print(f"({rep+1}/{strategy_repeats}) Only invalid results found, trying once more...")

    # repeat the strategy as specified
    repeated_results = list()
    mean_normalized_errors = list()
    total_time_results = np.array([])
    for rep in progressbar.progressbar(range(strategy['repeats']), redirect_stdout=True):
        attempt = 0
        only_invalid = True
        while only_invalid or (remove_duplicate_results and len_unique_res < max_num_evals):
            if attempt > 0:
                report_multiple_attempts(rep, len_res, len_unique_res, strategy['repeats'])
            res, total_time_ms = tune(kernel, kernel_name, device_name, strategy, tune_options, profiling)
            len_res = len(res)
            # check if there are only invalid configs, if so, try again
            only_invalid = len_res < 1 or min(res[:20], key=lambda x: x['time'])['time'] == 1e20
            unique_res = remove_duplicates(res, remove_duplicate_results)
            len_unique_res = len(unique_res)
            attempt += 1
        # register the results
        repeated_results.append(unique_res)
        total_time_results = np.append(total_time_results, total_time_ms)
        if len(strategy['nums_of_evaluations']) <= 0:
            nums_of_evaluations = np.append(nums_of_evaluations, len_unique_res)

    # gather profiling data and clear the profiler before the next round
    if profiling:
        stats = yappi.get_func_stats()
        # stats.print_all()
        path = "../experiments/profilings/random/profile-v2.prof"
        stats.save(path, type="pstat")    # pylint: disable=no-member
        yappi.clear_stats()

    # transpose and summarise to get the results per number of evaluations
    strategy['nums_of_evaluations'] = nums_of_evaluations
    results_to_write = deepcopy(expected_results)
    results_to_write = transpose_results(results_to_write, repeated_results, optimization, strategy)
    results_to_write = summarise_results(default_records, results_to_write, total_time_results, strategy, tune_options)
    return results_to_write


def transpose_results(results_to_write: dict, repeated_results: list, optimization: str, strategy: dict) -> dict:
    """ Transposes the results for summarise_results to go from num. of evaluations per result to results per num. of evaluations """
    nums_of_evaluations = strategy['nums_of_evaluations']
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
            loss = best['loss'] if 'loss' in best else np.nan
            noise = best['noise'] if 'noise' in best else np.nan

            # write to the results to the arrays
            result = results_to_write['results_per_number_of_evaluations'][str(num_of_evaluations)]
            result['actual_num_evals'] = np.append(result['actual_num_evals'], len(limited_res))
            result['time'] = np.append(result['time'], time)
            result['GFLOP/s'] = np.append(result['GFLOP/s'], gflops)
            result['loss'] = np.append(result['loss'], loss)
            result['noise'] = np.append(result['noise'], noise)
            result['cumulative_execution_time'] = np.append(result['cumulative_execution_time'], cumulative_execution_time)
    return results_to_write


def summarise_results(default_records: dict, expected_results: dict, total_time_results: np.ndarray, strategy: dict, tune_options: dict) -> dict:
    """ For every number of evaluations specified, find the best and collect details on it """

    # create the results dict for this strategy
    nums_of_evaluations = strategy['nums_of_evaluations']
    results_to_write = deepcopy(expected_results)

    # add the total time in miliseconds
    total_time_mean = np.mean(total_time_results)
    total_time_std = np.std(total_time_results)
    total_time_mean_per_eval = total_time_mean / nums_of_evaluations[-1]
    print("Total mean time: {} ms, std {}".format(round(total_time_mean, 3), round(total_time_std, 3)))
    results_to_write['total_time_mean'] = total_time_mean
    results_to_write['total_time_err'] = total_time_std

    for num_of_evaluations in nums_of_evaluations:
        result = results_to_write['results_per_number_of_evaluations'][str(num_of_evaluations)]

        # automatically summarise from default_records
        # TODO look into calculation of compile and execution times
        result['mean_cumulative_compile_time'] = 0
        cumulative_total_time = total_time_mean_per_eval * num_of_evaluations
        mean_runtimes = ['mean_cumulative_strategy_time', 'mean_cumulative_total_time']
        for key in default_records.keys():
            if key in mean_runtimes:
                continue
            if key == 'mean_cumulative_strategy_time':
                result[
                    'mean_cumulative_strategy_time'] = cumulative_total_time - result['mean_cumulative_compile_time'] - result['mean_cumulative_execution_time']
                if tune_options['simulation_mode']:
                    result['mean_cumulative_strategy_time'] = cumulative_total_time
            elif key == 'mean_cumulative_total_time':
                result['mean_cumulative_total_time'] = cumulative_total_time + result['mean_cumulative_compile_time'] + result['mean_cumulative_execution_time']
                if tune_options['simulation_mode']:
                    result['mean_cumulative_total_time'] = cumulative_total_time
            elif key == 'mean_cumulative_compile_time':
                continue
            elif key.startswith('mean_'):
                result[key] = np.mean(result[key.replace('mean_', '')])
            elif key.startswith('err_'):
                result[key] = np.std(result[key.replace('err_', '')])

        # summarise execution times
        # TODO do this properly

        # check for errors
        if 'err_actual_num_evals' in default_records.keys() and result['err_actual_num_evals'] != 0:
            raise ValueError('The number of actual evaluations over the runs has varied: {}'.format(result['actual_num_evals']))
        if 'mean_actual_num_evals' in default_records.keys() and result['mean_actual_num_evals'] != num_of_evaluations:
            print(
                "The set number of evaluations ({}) is not equal to the actual number of evaluations ({}). Try increasing the fraction or maxiter in strategy options."
                .format(num_of_evaluations, result['mean_actual_num_evals']))

    return results_to_write
