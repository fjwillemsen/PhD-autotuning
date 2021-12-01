""" Visualize the results of the experiments """
import argparse
import numpy as np
# import pandas as pd
# import statistics
from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# import seaborn as sns

from caching import CachedObject
from experiments2 import execute_experiment, create_expected_results

# The kernel information per device and device information for visualization purposes
marker_variatons = ['v', 's', '*', '1', '2', 'd', 'P', 'X']
kernel_device_info = {
    'A100': {
        'name': 'A100',
        'displayname': 'A100',
        'kernels': {
            'GEMM': {
                'absolute_optimum': 8.518111999999999,
                'y_axis_upper_limit': 9.9,
            },
            'convolution': {
                'absolute_optimum': 0.7390080038458109,
                'y_axis_upper_limit': 1.0,
            },
            'pnpoly': {
                'absolute_optimum': 13.090592086315155,
                'y_axis_upper_limit': 13.45,
            },
            'expdist': {
                'absolute_optimum': 33.87768375922341,
                'y_axis_upper_limit': 47,
            },
            'adding': {
                'absolute_optimum': 1.4682930000126362,
                'y_axis_upper_limit': None,
            },
        },
    },
    'RTX_2070_Super': {
        'name': 'RTX_2070_Super',
        'displayname': 'RTX 2070 Super',
        'kernels': {
            'GEMM': {
                'absolute_optimum': 17.111732999999997,
                'y_axis_upper_limit': 22.8,
            },
            'convolution': {
                'absolute_optimum': 1.2208920046687126,
                'y_axis_upper_limit': 1.9,
            },
            'pnpoly': {
                'absolute_optimum': 12.325379967689514,
                'y_axis_upper_limit': 13.5,
            },
        },
    },
    'GTX_TITAN_X': {
        'name': 'GTX_TITAN_X',
        'displayname': 'GTX Titan X',
        'kernels': {
            'GEMM': {
                'absolute_optimum': 28.307017000000002,
                'y_axis_upper_limit': 37,
            },
            'convolution': {
                'absolute_optimum': 1.6253190003335476,
                'y_axis_upper_limit': 2.52,
            },
            'pnpoly': {
                'absolute_optimum': 26.968406021595,
                'y_axis_upper_limit': 35.1,
            },
        },
    },
    'generator': {
        'name': 'generator',
        'displayname': 'Synthetic searchspaces',
        'kernels': {
            'Rosenbrock': {
                'absolute_optimum': 0.0,
                'y_axis_upper_limit': 4,
            },
            'Mishrasbird': {
                'absolute_optimum': 0.0,
                'y_axis_upper_limit': 51,
            },
            'Gomez-Levy': {
                'absolute_optimum': 0.0,
                'y_axis_upper_limit': 0.27,
            },
        }
    }
}


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


class Visualize():
    """ Class for visualization of experiments """

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

    def __init__(self, experiment_filename: str) -> None:
        self.experiment, self.strategies, self.caches = execute_experiment(experiment_filename, profiling=False)
        print("\n\n")

        # find the minimum and maximum number of evaluations over all strategies
        self.min_num_evals = np.Inf
        self.max_num_evals = 0
        for strategy in self.strategies:
            num_evals = strategy['nums_of_evaluations']
            self.min_num_evals = min(min(num_evals), self.min_num_evals)
            self.max_num_evals = max(max(num_evals), self.max_num_evals)

        # visualize
        for gpu_name in self.experiment['GPUs']:
            for kernel_name in self.experiment['kernels']:
                print(f"  visualizing {kernel_name} on {gpu_name}")

                # create the figure and plots
                fig, axs = plt.subplots(ncols=2, figsize=(20, 10))    # if multiple subplots, pass the axis to the plot function with axs[0] etc.
                title = f"{kernel_name} on {gpu_name}"
                fig.canvas.set_window_title(title)
                fig.suptitle(title)

                # prefetch the cached strategy results
                cache = self.caches[gpu_name][kernel_name]
                strategies_data = list()
                for strategy in self.strategies:
                    expected_results = create_expected_results(strategy)
                    cached_data = cache.get_strategy_results(strategy['name'], strategy['options'], strategy['repeats'], expected_results)
                    strategies_data.append(dict(cached_data))
                    if cached_data is None:
                        raise ValueError(f"Strategy {strategy['display_name']} not in cache, make sure execute_experiment() has ran first")

                # visualize the results
                info = kernel_device_info[gpu_name]['kernels'][kernel_name]
                self.plot_strategies_over_evals(axs[0], strategies_data, info, y_metric='time')
                self.plot_strategy_runtime_barchart(axs[1], strategies_data, info)

                # finalize the figure and display it
                fig.tight_layout()
                plt.show()

    def order_strategies(self, main_bar_group: str) -> list:
        """ Orders the strategies in the order we wish to have them plotted """
        if main_bar_group == '':
            return self.strategies
        strategies_ordered = [None] * len(self.strategies)
        index_first = 0
        index_last = len(strategies_ordered) - 1
        for strategy in self.strategies:
            if 'bar_group' not in strategy:
                strategies_ordered = self.strategies
                break
            if strategy['bar_group'] == main_bar_group:
                strategies_ordered[index_last] = strategy
                index_last -= 1
            else:
                strategies_ordered[index_first] = strategy
                index_first += 1
        if None in strategies_ordered:
            strategies_ordered = self.strategies
        return strategies_ordered

    def plot_strategies_over_evals(self, ax: plt.Axes, strategies_data: list, info: dict, x_metric='num_evals', y_metric='GFLOP/s', shaded=True,
                                   plot_errors=True, main_bar_group=''):
        """ Plots all strategies with errorbars, shaded plots a shaded error region instead of error bars. Y-axis and X-axis metrics can be chosen. """

        # plotting setup
        bar_groups_markers = {
            'reference': '.',
            'basic': '+',
            'multi': 'd',
            'default': '+',
            'pruned': 'd',
            'bo': 'D',
        }
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        strategies_ordered = self.order_strategies(main_bar_group)

        # plot absolute optimum
        absolute_optimum = info['absolute_optimum']
        if absolute_optimum is not None:
            ax.plot([self.min_num_evals, self.max_num_evals], [absolute_optimum, absolute_optimum], linestyle='-',
                    label="True optimum {}".format(round(absolute_optimum, 3)), color='black')

        # plotting
        for strategy_index, strategy in enumerate(strategies_ordered):
            color = colors[strategy_index]
            collect_xticks = list()

            # use the results to draw the plot
            results = strategies_data[strategy_index]['results']['results_per_number_of_evaluations']
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
            marker = 'o'
            alpha = 1.0
            fill_alpha = 0.2
            plot_error = plot_errors
            # TODO change below to reduce cognitive complexity
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

        # set the y-axis limit
        y_axis_lower_limit = absolute_optimum - ((ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01)
        ax.set_ylim(y_axis_lower_limit, ax.get_ylim()[1])
        if 'y_axis_upper_limit' in info:
            y_axis_upper_limit = info['y_axis_upper_limit']
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

    def plot_strategy_runtime_barchart(self, ax: plt.Axes, strategies_data: list, info: dict, main_bar_group=''):
        strategies_ordered = self.order_strategies(main_bar_group)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        x_axis = list()
        total_times_mean = np.array([])
        total_times_err = np.array([])
        for strategy_index, strategy in enumerate(strategies_ordered):
            x_axis.append(strategy['display_name'])
            repeats = strategy['repeats']
            results = strategies_data[strategy_index]['results']
            miliseconds_to_seconds = lambda ms: ms / 1000
            total_times_mean = np.append(total_times_mean, miliseconds_to_seconds(results['total_time_mean']) / repeats)
            total_times_err = np.append(total_times_err, miliseconds_to_seconds(results['total_time_err']) / repeats)

        ax.bar(x_axis, total_times_mean, color=colors)
        ax.errorbar(x_axis, total_times_mean, yerr=total_times_err, fmt="o", color='r')
        ax.set_ylabel('Time in seconds per full strategy run')


if __name__ == "__main__":
    CLI = argparse.ArgumentParser()
    CLI.add_argument("experiment", type=str, help="The experiment.json to execute, see experiments/template.json")
    args = CLI.parse_args()
    filepath = args.experiment
    if filepath is None:
        raise ValueError("Invalid '-experiment' option. Run 'visualize_experiments.py -h' to read more about the options.")

    Visualize(filepath)
