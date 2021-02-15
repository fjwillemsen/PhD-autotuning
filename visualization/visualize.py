import importlib
import os
os.chdir("../cached_runs")
import sys
sys.path.append("../cached_runs")
import json
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy

convolution = importlib.import_module("convolution")


class CachedObject():

    def __init__(self, kernel_name: str, device_name: str, strategies: dict = None):
        try:
            cache = CacheInterface.read(kernel_name, device_name)
            self.kernel_name = cache['kernel_name']
            self.device_name = cache['device_name']
            self.obj = cache
        except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
            print(e)
            self.kernel_name = kernel_name
            self.device_name = device_name
            self.obj = {
                "kernel_name": kernel_name,
                "device_name": device_name,
                "strategies": strategies
            }

    def read(self):
        return CacheInterface.read(self.kernel_name, self.device_name)

    def write(self):
        return CacheInterface.write(self.obj)

    # def build_strategy_index(self):
    #     index = dict()
    #     for i, strategy in enumerate(self.obj['strategies']):
    #         index[strategy['name']] = i
    #     return index

    def has_strategy(self, strategy_name: str) -> bool:
        return strategy_name in self.obj["strategies"].keys()

    def has_matching_strategy(self, strategy_name: str, iterations_name: str, iterations: [], repeats: int) -> bool:
        if self.has_strategy(strategy_name):
            strategy = self.obj['strategies'][strategy_name]
            if (strategy['name'] == strategy_name and strategy['iterations_name'] == iterations_name and strategy['iterations'] == iterations
                    and strategy['repeats'] == repeats):
                return True
        return False

    def get_strategy(self, strategy_name: str, iterations_name: str, iterations: [], repeats: int):
        if self.has_matching_strategy(strategy_name, iterations_name, iterations, repeats):
            return self.obj['strategies'][strategy_name]
        return None

    def set_strategy(self, strategy: dict(), gflops, gflops_error, evaluations):
        strategy_name = strategy['name']
        # delete old strategy if any
        if self.has_strategy(strategy['name']):
            del self.obj["strategies"][strategy_name]
        # set new strategy
        self.obj["strategies"][strategy_name] = strategy
        # set new values
        self.obj["strategies"][strategy_name]["gflops"] = gflops
        self.obj["strategies"][strategy_name]["gflops_error"] = gflops_error
        self.obj["strategies"][strategy_name]["evaluations"] = evaluations
        self.write()


class CacheInterface:

    def file_n(kernel_name: str, device_name: str) -> str:
        return 'cached_plot_' + kernel_name + '_' + device_name + '.json'

    def read(kernel_name: str, device_name: str) -> CachedObject:
        filename = CacheInterface.file_n(kernel_name, device_name)
        with open(filename) as json_file:
            return json.load(json_file)

    def write(cached_object: dict):
        filename = CacheInterface.file_n(cached_object['kernel_name'], cached_object['device_name'])
        with open(filename, 'w') as json_file:
            json.dump(cached_object, json_file)


def plot(kernel_name='convolution', device_name='RTX_2070_SUPER', repeats=3):
    total_iterations = 9400
    iteration_fractions = np.arange(0.001, 0.2, 0.01)
    iteration_fractions_list = iteration_fractions.tolist()
    iterations_list = (total_iterations * iteration_fractions).astype(int).tolist()

    strategies = {
        'brute_force': {
            'name': 'brute_force',
            'display_name': 'Brute Force',
            'iterations_name': 'maxiter',
            'iterations': [total_iterations],
            'repeats': 1
        },
        'random_sample': {
            'name': 'random_sample',
            'display_name': 'Random Sample',
            'iterations_name': 'fraction',
            'iterations': iteration_fractions_list,
            'repeats': 10
        },
    # 'bayes_opt': {
    #     'name': 'bayes_opt',
    #     'display_name': 'Bayesian Optimization',
    #     'iterations_name': 'maxiter',
    #     'iterations': iterations_list,
    #     'repeats': 3
    # },
        'firefly': {
            'name': 'firefly_algorithm',
            'display_name': 'Firefly algorithm',
            'iterations_name': 'maxiter',
            'iterations': iterations_list,
            'repeats': 3
        },
        'pso': {
            'name': 'pso',
            'display_name': 'Particle Swarm Optimization',
            'iterations_name': 'maxiter',
            'iterations': iterations_list,
            'repeats': 3
        }
    }

    # strategies = {
    #     'brute_force': {
    #         'name': 'brute_force',
    #         'display_name': 'Brute Force',
    #         'iterations_name': 'maxiter',
    #         'iterations': [total_iterations],
    #         'repeats': 1
    #     },
    #     'random_sample': {
    #         'name': 'random_sample',
    #         'display_name': 'Random Sample',
    #         'iterations_name': 'fraction',
    #         'iterations': iteration_fractions_list,
    #         'repeats': 10
    #     }
    # }

    # get or create a cache
    cache = CachedObject(kernel_name, device_name, deepcopy(strategies))

    # run all strategies
    for strategy in strategies.values():
        print("Running {}, iterations:".format(strategy['display_name']), end=' ')
        evaluations = np.array([])
        gflops = np.array([])
        gflops_error = np.array([])

        # if the strategy is in the cache, use cached data
        cached_data = cache.get_strategy(strategy['name'], strategy['iterations_name'], strategy['iterations'], strategy['repeats'])
        if cached_data is not None and all(key in cached_data for key in ("evaluations", "gflops", "gflops_error")):
            evaluations = np.array(cached_data['evaluations'])
            gflops = np.array(cached_data['gflops'])
            gflops_error = np.array(cached_data['gflops_error'])
            # add to the plot
            plt.errorbar(cached_data['evaluations'], cached_data['gflops'], cached_data['gflops_error'], marker='o', label=strategy['display_name'])
            print("Retrieved from cache")
            continue

        # execute the strategy
        for iteration in strategy['iterations']:
            full_iteration = round(iteration * total_iterations, 0) if strategy['iterations_name'] == 'fraction' else iteration
            print("{}".format(full_iteration), end=', ', flush=True)
            strategy_options = {
                strategy['iterations_name']: iteration
            }
            res = None
            gflops_local = np.array([])
            for _ in range(strategy['repeats']):
                # print(rep + 1, end=' ')
                res, _ = convolution.tune(device_name=device_name, strategy=strategy['name'], strategy_options=strategy_options, verbose=False, quiet=True)
                best = min(res, key=lambda x: x['time'])
                gflops_local = np.append(gflops_local, best['GFLOP/s'])

            # register the average values
            gflops = np.append(gflops, np.mean(gflops_local))
            gflops_error = np.append(gflops_error, np.std(gflops_local))
            evaluations = np.append(evaluations, full_iteration)
            # try:
            #     assert full_iteration >= len(res)
            # except:
            #     raise AssertionError("Number of executed evaluations ({}) exceeds maximum requested evaluations ({})".format(len(res), evaluations[-1]))

        # add to the plot
        plt.errorbar(evaluations, gflops, gflops_error, marker='o', label=strategy['display_name'])

        # write to the cache
        cache.set_strategy(strategy=deepcopy(strategy), gflops=gflops.tolist(), gflops_error=gflops_error.tolist(), evaluations=evaluations.tolist())
        print("")

    # plot setup
    # plt.xlim(right=10000)
    plt.xlabel("Number of evaluations required")
    plt.ylabel("GFLOP/s")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 1 and len(sys.argv) != 3:
        print("Usage: ./visualize.py [kernel name] [device name]")
        exit(1)

    if len(sys.argv) > 1:
        kernel_name = sys.argv[1] or None
        device_name = sys.argv[2] or None
        plot(kernel_name, device_name=device_name)
    else:
        plot()
