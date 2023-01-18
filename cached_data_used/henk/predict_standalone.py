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

def predict(partitioning, df_diff, df_start, df_last):
    result = []
    start = 0

    for gpu, layers in enumerate(partitioning):
        start_mem = 0
        
        if start == 0:# corner case
            start_mem = df_diff.loc[df_diff['layer'] == str(start)]
        else:
            start_mem = df_start.loc[df_start['layer'] == str(start)]
        
        start_mem = start_mem.values.tolist()
        if (len(start_mem) == 0):
            start_mem = 0.0
        else:
            start_mem = float(start_mem[0][0])

        for i in range(start+1, start+layers):
            mem = df_diff.loc[df_diff['layer'] == str(i)]
            mem = mem.values.tolist()
            if (len(mem) == 0):
                mem = 0.0
            else:
                mem = float(mem[0][0])

            start_mem += float(mem)

        start += layers
        result.append(start_mem)

    return result

def generate_partitionings_recursive(n_layers, n_gpus, partial_result):
    if n_gpus == 1:
        yield partial_result + [n_layers]
        return
    for i in range(1, n_layers - n_gpus + 2):
        yield from generate_partitionings_recursive(n_layers - i, n_gpus - 1, partial_result + [i])

def generate_partitionings(n_layers, n_gpus):
    yield from generate_partitionings_recursive(n_layers, n_gpus, [])

def predict_peak_mem(per_layer_file, layer_diff_file, last_layer_file, results_out_file=None, n_layers=30, n_gpus=4, df_results_file=None):
    if df_results_file is not None:
        df_results = pd.read_pickle(df_results_file)
        return df_results

    else:
        # Read data (as csv), convert to dataframe and turn into numeric values.
        df_start = read_data(per_layer_file, None)
        df_diff = read_data(layer_diff_file, None)
        df_last = read_data(last_layer_file, None)
        # print(df_start)
        # print(df_diff)
        # print(df_last)

        start = time.time()
        results = []
        best_result = math.inf
        print("Partitioning, prediction")
        for partitioning in generate_partitionings(n_layers, n_gpus):
            prediction = predict(partitioning, df_diff, df_start, df_last)
            # print(partitioning, prediction)
            results.append((partitioning, prediction))
        
        end = time.time()
        print(end-start)
        print("done")
        df_results = pd.DataFrame(results, columns=['partitioning', 'prediction'])
        
        if results_out_file is not None:
            df_results.to_pickle(results_out_file)

        return df_results

if __name__ == "__main__":
    n_layers = 30
    n_gpus = 4
    # In- and output file names
    df_results_file = "results.pkl"
    per_layer_file = "pl.csv"
    per_layer_diff_file = "pl_diff.csv"
    last_layer_file = "ll.csv"

    n_layers = 42
    n_gpus = 4
    # In- and output file names
    df_results_file = "results_amoebanet.pkl"
    per_layer_file = "pl_amoebanet.csv"
    per_layer_diff_file = "pl_diff_amoebanet.csv"
    last_layer_file = "ll_amoebanet.csv"
    
    pred_df = predict_peak_mem(per_layer_file, per_layer_diff_file, last_layer_file, df_results_file, n_layers=n_layers, n_gpus=n_gpus, df_results_file=None)

    print(pred_df)

    # Take max of predictions
    pred_df['peak_prediction'] = pred_df.prediction.apply(max)

    print("Partitionings with the lowest predicted peak memory usage:")
    print(pred_df[pred_df.peak_prediction == pred_df.peak_prediction.min()])
