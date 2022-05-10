import sys
import csv
import pandas as pd


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

def predict_peak_mem_kernel(gpu1, gpu2, gpu3, gpu4):
    global df_diff
    global df_start
    global df_last
    global global_n_layers

    layers_gpu_0, layers_gpu_1, layers_gpu_2, layers_gpu_3 = gpu1, gpu2, gpu3, gpu4

    partitioning = [int(layers_gpu_0), int(layers_gpu_1), int(layers_gpu_2), int(layers_gpu_3)]

    # Check if parameters within bounds (sum equal to n_layers). Return a high values if
    # not within bounds so that this part of the search space is unlikely to be explored
    # further.
    if sum(partitioning) != global_n_layers:
        raise ValueError("Sum not equal to total number of layers")
        # return 1000

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

    # Return peak memory usage across all GPUs.
    return max(result)



# Global Setup
possible_neural_networks = ["vgg11", "amoebanet"]
neural_networks = "amoebanet"

if neural_networks == "vgg11":
    n_layers = 30
    n_gpus = 4
    # In- and output file names
    df_results_file = "results.pkl"
    per_layer_file = "pl.csv"
    layer_diff_file = "pl_diff.csv"
    last_layer_file = "ll.csv"
elif neural_networks == "amoebanet":
    n_layers = 42
    n_gpus = 4
    # In- and output file names
    df_results_file = "results_amoebanet.pkl"
    per_layer_file = "pl_amoebanet.csv"
    layer_diff_file = "pl_diff_amoebanet.csv"
    last_layer_file = "ll_amoebanet.csv"
else:
    print("Invalid arguments.")
    sys.exit(0)

global_n_layers = n_layers

# Read data (as csv), convert to dataframe and turn into numeric values.
df_start = read_data(per_layer_file, None)
df_diff = read_data(layer_diff_file, None)
df_last = read_data(last_layer_file, None)
