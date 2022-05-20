""" Visualize the results of the experiments """
import argparse
from collections import defaultdict
import numpy as np
from copy import deepcopy
from typing import Tuple
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import auc

from caching import CachedObject
from experiments2 import execute_experiment, create_expected_results

import sys

sys.path.append("..")
# TODO from cached_data_used.kernel_info_generator import kernels_device_info # check whether this is necessary

# read the kernel info dictionary from file
import json
with open("../cached_data_used/kernel_info.json") as file:
    kernels_device_info_data = file.read()
kernels_device_info = json.loads(kernels_device_info_data)

# The kernel information per device and device information for visualization purposes
marker_variatons = ['v', 's', '*', '1', '2', 'd', 'P', 'X']


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

def smoothing_filter(array: np.ndarray, window_length: int) -> np.ndarray:
    """ Create a rolling average where the kernel size is the smoothing factor """
    from scipy.signal import savgol_filter
    return savgol_filter(array, window_length, 3)


class Visualize():
    """ Class for visualization of experiments """

    x_metric_displayname = dict({
        'num_evals': 'Number of function evaluations used',
        'strategy_time': 'Average time taken by strategy in miliseconds',
        'compile_time': 'Average compile time in miliseconds',
        'execution_time': 'Evaluation execution time taken in miliseconds',
        'total_time': 'Average total time taken in miliseconds',
        'kerneltime': 'Total kernel compilation and runtime in seconds',
    })

    y_metric_displayname = dict({
        'objective': 'Best found objective function value',
        'objective_baseline': 'Best found objective function value relative to baseline',
        'time': 'Best found kernel time in miliseconds',
        'GFLOP/s': 'GFLOP/s',
    })

    def __init__(self, experiment_filename: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.experiment, self.strategies, self.caches = execute_experiment(experiment_filename, profiling=False, kernel_info_stats=kernels_device_info)
        print("\n\n")

        # find the minimum and maximum number of evaluations over all strategies
        self.min_num_evals = np.Inf
        self.max_num_evals = 0
        for strategy in self.strategies:
            num_evals = strategy['nums_of_evaluations']
            self.min_num_evals = min(min(num_evals), self.min_num_evals)
            self.max_num_evals = max(max(num_evals), self.max_num_evals)

        # visualize
        cutoff_quantile = self.experiment['cutoff_quantile']
        all_strategies_curves = list()
        for gpu_name in self.experiment['GPUs']:
            for kernel_name in self.experiment['kernels']:
                print(f"  visualizing {kernel_name} on {gpu_name}")

                # create the figure and plots
                fig, axs = plt.subplots(ncols=1, figsize=(15, 8))    # if multiple subplots, pass the axis to the plot function with axs[0] etc.
                if not isinstance(axs, list):
                    axs = [axs]
                title = f"{kernel_name} on {gpu_name}"
                fig.canvas.manager.set_window_title(title)
                fig.suptitle(title)

                # prefetch the cached strategy results
                cache = self.caches[gpu_name][kernel_name]
                strategies_data = list()
                for strategy in self.strategies:
                    expected_results = create_expected_results()
                    cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                    if cached_data is None:
                        raise ValueError(f"Strategy {strategy['display_name']} not in cache, make sure execute_experiment() has ran first")
                    strategies_data.append(cached_data)

                # visualize the results
                info = kernels_device_info[gpu_name]['kernels'][kernel_name]
                subtract_baseline = self.experiment['relative_to_baseline']
                strategies_curves = self.get_strategies_curves(strategies_data, info, cutoff_quantile, subtract_baseline=subtract_baseline)
                self.plot_strategies_curves(axs[0], strategies_data, strategies_curves, info, cutoff_quantile, subtract_baseline=subtract_baseline)
                all_strategies_curves.append(strategies_curves)

                # finalize the figure and display it
                fig.tight_layout()
                plt.show()

        print("\n")
        for strategy_index, strategy in enumerate(self.strategies):
            perf = list()
            for strategies_curves in all_strategies_curves:
                for strategy_curve in strategies_curves['strategies']:
                    if strategy_curve['strategy_index'] == strategy_index:
                        perf.append(strategy_curve['performance'])
            print(f"{strategy['display_name']} performance across kernels: {np.mean(perf)}")

    def get_strategies_curves(self, strategies_data: list, info: dict, cutoff_quantile: float, subtract_baseline=True, main_bar_group='') -> dict:
        # get the baseline
        baseline_result = strategies_data[0]['results']
        x_axis = np.array(baseline_result['interpolated_time'])
        y_axis_baseline = np.array(baseline_result['interpolated_objective'])

        # find the cutoff point
        # the cutoff point is the time it takes the baseline to find <= [quantile]% of the absolute optimum
        sorted_objective_values = np.array(info['sorted_times'])
        objective_value_at_quantile = np.quantile(sorted_objective_values, 1-cutoff_quantile)   # sorted in ascending order, so inverse quantile
        try:
            cutoff_point = np.argwhere(y_axis_baseline <= objective_value_at_quantile)[0]
            assert cutoff_point == int(cutoff_point)    # ensure that it is an integer
            cutoff_point = int(cutoff_point)
        except IndexError:
            raise ValueError(f"The baseline has not reliably found the {cutoff_quantile} cutoff quantile, either decrease the cutoff or increase the allowed time.")
        # print(f"Percentage of baseline search space used to get {cutoff_quantile} cutoff quantile: {(cutoff_point / y_axis_baseline.size)*100}%")
        x_axis = x_axis[:cutoff_point]
        y_axis_baseline = y_axis_baseline[:cutoff_point]

        # create resulting dict
        strategies_curves = dict({
            'baseline': {
                'x_axis': x_axis,
                'y_axis': y_axis_baseline
            },
            'strategies': list()
        })

        performances = list()
        for strategy_index, strategy in enumerate(self.strategies):
            if 'hide' in strategy.keys() and strategy['hide']:
                continue

            # get the data
            strategy = strategies_data[strategy_index]
            results = strategy['results']
            y_axis = np.array(results['interpolated_objective'])[:cutoff_point]
            y_axis_std = np.array(results['interpolated_objective_std'])[:cutoff_point]
            y_axis_std_lower = y_axis_std
            y_axis_std_upper = y_axis_std

            # find out where the global optimum is found and substract the baseline
            # found_opt = np.argwhere(y_axis == absolute_optimum)
            if subtract_baseline:
                y_axis =  y_axis - y_axis_baseline

            # quantify the performance of this strategy
            if subtract_baseline:
                # use average distance from random
                performance = np.mean(y_axis)
            else:
                # use area under curve approach
                performance = auc(x_axis, y_axis)
            performances.append(performance)
            print(f"Performance of {strategy['display_name']}: {performance}")

            # write to resulting dict
            result_dict = dict({
                'strategy_index': strategy_index,
                'y_axis': y_axis,
                'y_axis_std': y_axis_std,
                'y_axis_std_lower': y_axis_std_lower,
                'y_axis_std_upper': y_axis_std_upper,
                'performance': performance,
            })
            strategies_curves['strategies'].append(result_dict)

        print(f"Mean performance across strategies: {np.mean(performances)}")   # the higher the mean, the easier a search space is for the baseline
        return strategies_curves


    def plot_strategies_curves(self, ax: plt.Axes, strategies_data: list, strategies_curves: dict, info: dict, shaded=True, plot_errors=False, subtract_baseline=True):
        """ Plots all optimization strategy curves """
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        bar_groups_markers = {
            'reference': '.',
            'old': '+',
            'new': 'd'
        }

        baseline = strategies_curves['baseline']
        x_axis = baseline['x_axis']

        # plot the absolute optimum
        absolute_optimum = info['absolute_optimum']
        if subtract_baseline is False and absolute_optimum is not None:
            ax.plot([x_axis[0], x_axis[-1]], [absolute_optimum, absolute_optimum], linestyle='-',
                    label="True optimum {}".format(round(absolute_optimum, 3)), color='black')

        color_index = 0
        marker = ','
        for strategy_curves in strategies_curves['strategies']:
            # get the data
            strategy = strategies_data[strategy_curves['strategy_index']]
            y_axis = strategy_curves['y_axis']
            y_axis_std = strategy_curves['y_axis_std']
            y_axis_std_lower = strategy_curves['y_axis_std_lower']
            y_axis_std_upper = strategy_curves['y_axis_std_upper']

            # set colors, transparencies and markers
            color = colors[color_index]
            color_index += 1
            alpha = 1.0
            fill_alpha = 0.2
            if 'bar_group' in strategy:
                bar_group = strategy['bar_group']
                marker = bar_groups_markers[bar_group]

            # plot the data
            if shaded is True:
                if plot_errors:
                    ax.fill_between(x_axis, y_axis_std_lower, y_axis_std_upper, alpha=fill_alpha, antialiased=True, color=color)
                ax.plot(x_axis, y_axis, marker=marker, alpha=alpha, linestyle='-', label=f"{strategy['display_name']}", color=color)
            else:
                if plot_errors:
                    ax.errorbar(x_axis, y_axis, y_axis_std, marker=marker, alpha=alpha, linestyle='--', label=strategy['display_name'])
                else:
                    ax.plot(x_axis, y_axis, marker=marker, linestyle='-', label=f"{strategy['display_name']}", color=color)

        # finalize plot
        ax.set_xlabel(self.x_metric_displayname['kerneltime'])
        ax.set_ylabel(self.y_metric_displayname['objective_baseline' if subtract_baseline else 'objective'])
        ax.legend()
        if plot_errors is False:
            ax.grid(axis='y', zorder=0, alpha=0.7)


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("experiment", type=str, help="The experiment.json to execute, see experiments/template.json")
    args = CLI.parse_args()
    filepath = args.experiment
    if filepath is None:
        raise ValueError("Invalid '-experiment' option. Run 'visualize_experiments.py -h' to read more about the options.")

    Visualize(filepath)
