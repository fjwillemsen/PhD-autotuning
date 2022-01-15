""" Main experiments code """

import argparse
from importlib import import_module
import json
import os
import sys
from typing import Tuple
import pathvalidate
from copy import deepcopy
import numpy as np

from runner import collect_results
from caching import CachedObject
from record_data import default_records


def change_directory(path: str):
    os.chdir(path)
    sys.path.append(path)


def get_experiment(filename: str) -> dict:
    """ Gets the experiment from the .json file """
    folder = 'experiments/'
    extension = '.json'
    if not filename.endswith(extension):
        filename = filename + extension
    path = filename
    if not filename.startswith(folder):
        path = folder + filename
    path = pathvalidate.sanitize_filepath(path)
    with open(path) as file:
        experiment = json.load(file)
        return experiment


def get_strategies(experiment: dict) -> dict:
    """ Gets the strategies from an experiments file by augmenting it with the defaults """
    strategy_defaults = experiment['strategy_defaults']
    strategies = experiment['strategies']
    for strategy in strategies:
        for default in strategy_defaults:
            if not default in strategy:
                strategy[default] = strategy_defaults[default]
        if not 'min_nums_of_evaluations' in strategy:
            strategy['min_nums_of_evaluations'] = min(strategy['nums_of_evaluations'])
        if not 'max_nums_of_evaluations' in strategy:
            strategy['max_nums_of_evaluations'] = max(strategy['nums_of_evaluations'])
    return strategies


def create_expected_results(strategy: dict) -> dict:
    """ Creates a dict to put the expected results into """
    nums_of_evaluations = strategy['nums_of_evaluations']
    if len(nums_of_evaluations) <= 0:
        raise ValueError("No evaluations to perform")

    list_nums_of_evaluations = np.array(nums_of_evaluations).astype(int)
    list_nums_of_evaluations_as_keys = list_nums_of_evaluations.astype(str).tolist()

    # fill the results with the default values
    expected_results = dict({
        'results_per_number_of_evaluations': dict.fromkeys(list_nums_of_evaluations_as_keys),
        'total_time_mean': None,
        'total_time_err': None,
    })
    for num_of_evaluations in list_nums_of_evaluations_as_keys:
        expected_results['results_per_number_of_evaluations'][num_of_evaluations] = deepcopy(default_records)
    return expected_results


def execute_experiment(filepath: str, profiling: bool) -> Tuple[dict, dict, dict]:
    """ Executes the experiment by retrieving it from the cache or running it """
    experiment = get_experiment(filepath)
    print(f"Starting experiment \'{experiment['name']}\'")
    change_directory("../cached_data_used")
    strategies = get_strategies(experiment)
    kernel_names = experiment['kernels']
    kernels = list(import_module(kernel_name) for kernel_name in kernel_names)

    # execute each strategy in the experiment per GPU and kernel
    caches = dict()
    for gpu_name in experiment['GPUs']:
        caches[gpu_name] = dict()
        for index, kernel in enumerate(kernels):
            kernel_name = kernel_names[index]
            print(f"  running {kernel_name} on {gpu_name}")
            # get or create a cache to write the results to
            cache = CachedObject(kernel_name, gpu_name, deepcopy(strategies))
            for strategy in strategies:
                print(f"    | with strategy {strategy['display_name']}")
                # if the strategy is in the cache, use cached data
                expected_results = create_expected_results(strategy)
                if 'ignore_cache' not in strategy:
                    cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                    if cached_data is not None:
                        print("| retrieved from cache")
                        continue
                # execute each strategy that is not in the cache
                strategy_results = collect_results(kernel, kernel_name, gpu_name, strategy, expected_results, default_records, profiling)
                # write the results to the cache
                cache.set_strategy(deepcopy(strategy), strategy_results)
            caches[gpu_name][kernel_name] = cache

    return experiment, strategies, caches


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("experiment", type=str, help="The experiment.json to execute, see experiments/template.json")
    args = CLI.parse_args()
    experiment_filepath = args.experiment
    if experiment_filepath is None:
        raise ValueError("Invalid '-experiment' option. Run 'experiments.py -h' to read more about the options.")

    execute_experiment(experiment_filepath, profiling=False)
