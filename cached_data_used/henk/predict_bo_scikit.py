import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import math
import skopt

def read_data(filename, out_filename):
    # Open file and read data
    rows = []
    with open(filename) as csv_file:
        read_csv = csv.reader(csv_file, delimiter=',')
        header = next(read_csv)
        for row in read_csv:
            rows.append(row)

    # Transform to dataframe
    df = pd.DataFrame(rows)
    df.columns = header
    return df

def percent_to_layers(params, n_layers):
    remaining_layers = n_layers
    new_params = []
    n_gpus = len(params) + 1

    # Convert percentage of the remaining layers (in range [1, 100]) to a discrete
    # number (number of layers as an integer).
    for gpu_id, percentage in enumerate(params):
        layers = round((percentage / 100.0 * (remaining_layers - (n_gpus - (gpu_id + 1)))))
        # Minimum of 1 layer
        if layers == 0:
            layers = 1
        remaining_layers = remaining_layers - layers
        new_params.append(layers)
        # print(percentage, "\% of", remaining_layers, "is", layers, "sum equals", sum(new_params))

    # Last GPU gets remaining layers
    new_params.append(remaining_layers)

    # print(params, "becomes", new_params, "sum equals", sum(new_params))
    return new_params

def predict_bo(params):
    global df_diff
    global df_start
    global df_last
    global global_n_layers

    params = percent_to_layers(params, global_n_layers)
    layers_gpu_0, layers_gpu_1, layers_gpu_2, layers_gpu_3 = params

    # The bayesian optimization package does not support integer parameter values, so it
    # gives floats. Convert to integers first.
    # This approach is recommended in: https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
    partitioning = [int(layers_gpu_0), int(layers_gpu_1), int(layers_gpu_2), int(layers_gpu_3)]

    # Check if parameters within bounds (sum equal to n_layers). Return a high values if
    # not within bounds so that this part of the search space is unlikely to be explored
    # further.
    if sum(partitioning) != global_n_layers:
        return 1000

    result = []
    start = 0

    # Predict peak memory usage for each GPU.
    for gpu, layers in enumerate(partitioning):
        start_mem = 0

        # First determine memory usage caused by first layer placed on this GPU.
        if start == 0:# corner case
            start_mem = df_diff.loc[df_diff['layer'] == str(start)]
        else:
            start_mem = df_start.loc[df_start['layer'] == str(start)]

        # Convert to float value.
        start_mem = start_mem.values.tolist()
        if (len(start_mem) == 0):
            start_mem = 0.0
        else:
            start_mem = float(start_mem[0][0])

        # Add memory usage caused by other layers placed on this GPU.
        for i in range(start+1, start+layers):
            mem = df_diff.loc[df_diff['layer'] == str(i)]
            mem = mem.values.tolist()
            # Convert to float value.
            if (len(mem) == 0):
                mem = 0.0
            else:
                mem = float(mem[0][0])

            start_mem += mem

        start += layers
        result.append(start_mem)

    # Return peak memory usage across all GPUs. Invert the result because we use
    # optimizer.maximize but we are looking for the minimum.
    return max(result)

df_start = None
df_diff = None
df_last = None
global_n_layers = None

def predict_peak_mem(per_layer_file, layer_diff_file, last_layer_file, results_out_file=None, n_layers=30, n_gpus=4, df_results_file=None):
    global df_start
    global df_diff
    global df_last
    global global_n_layers

    global_n_layers = n_layers

    # Read data (as csv), convert to dataframe and turn into numeric values.
    df_start = read_data(per_layer_file, None)
    df_diff = read_data(layer_diff_file, None)
    df_last = read_data(last_layer_file, None)
    # print(df_start)
    # print(df_diff)
    # print(df_last)

    acq_func_kwargs = {"xi": 1000, "kappa": 0.01}

    # Parameter space: a value between 1 and 100 indicating how many of the remaining layers are placed
    # on a GPU (as a percentage).
    opt = skopt.Optimizer([(1, 100), (1, 100), (1, 100)],
        "GP",
        n_initial_points=30,
        # acq_func="EI",
         acq_optimizer="sampling",
         acq_func_kwargs=acq_func_kwargs)

    opt.run(predict_bo, n_iter=75)

    print(opt.get_result())
    print("Result:", opt.get_result().x, percent_to_layers(opt.get_result().x, global_n_layers), predict_bo(opt.get_result().x))
    print("Absolute best AmoebaNet (predicted): [27, 45, 53] [11, 13, 9, 9] 6.392")

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <neural network>")
        print("Possible values for neural network: vgg11 or amoebanet")
        sys.exit()

    if sys.argv[1] == "vgg11":
        n_layers = 30
        n_gpus = 4
        # In- and output file names
        df_results_file = "results.pkl"
        per_layer_file = "pl.csv"
        per_layer_diff_file = "pl_diff.csv"
        last_layer_file = "ll.csv"
    elif sys.argv[1] == "amoebanet":
        n_layers = 42
        n_gpus = 4
        # In- and output file names
        df_results_file = "results_amoebanet.pkl"
        per_layer_file = "pl_amoebanet.csv"
        per_layer_diff_file = "pl_diff_amoebanet.csv"
        last_layer_file = "ll_amoebanet.csv"
    else:
        print("Invalid arguments.")
        sys.exit(0)

    predict_peak_mem(per_layer_file, per_layer_diff_file, last_layer_file, df_results_file, n_layers=n_layers, n_gpus=n_gpus, df_results_file=None)
