import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import math

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

def predict_bo(layers_gpu_0, layers_gpu_1, layers_gpu_2, layers_gpu_3):
    global df_diff
    global df_start
    global df_last
    global global_n_layers

    # The bayesian optimization package does not support integer parameter values, so it
    # gives floats. Convert to integers first.
    # This approach is recommended in: https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
    partitioning = [int(layers_gpu_0), int(layers_gpu_1), int(layers_gpu_2), int(layers_gpu_3)]

    # Check if parameters within bounds (sum equal to n_layers). Return a high values if
    # not within bounds so that this part of the search space is unlikely to be explored
    # further.
    if sum(partitioning) != global_n_layers:
        return -1000

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

            start_mem += float(mem)

        start += layers
        result.append(start_mem)

    # Return peak memory usage across all GPUs. Invert the result because we use
    # optimizer.maximize but we are looking for the minimum.
    return -max(result)

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

    # Bayesian optimization:
    from bayes_opt import BayesianOptimization
    
    # Bounded region of parameter space
    # These parameters indicate the number of layers placed on each GPU (1 parameter per GPU).
    # The issue is that the sum must always be equal to the total number of layers of the neural
    # network (n_layers). I cannot enforce that restriction here, so the predict_bo function performs
    # a check on that and returns a very high value if it is not met.
    pbounds = {'layers_gpu_0': (1, 28), 'layers_gpu_1': (1, 27), 'layers_gpu_2': (1, 26), 'layers_gpu_3': (1, 25)}

    # Simple Bayesian optimization
    optimizer = BayesianOptimization(
        f=predict_bo,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=100,
        n_iter=20,
    )

    print(optimizer.max)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    return

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
