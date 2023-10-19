import json
from pathlib import Path

path_to_cachefiles = "../kernel_tuner_tutorial/energy/data"
input_file = Path(path_to_cachefiles, "GEMM_NVML_NVIDIA_A100-PCIE-40GB_freq_cache.json")
output_file = Path(
    path_to_cachefiles, "GEMM_NVML_NVIDIA_A100-PCIE-40GB_freq_cache_new.json"
)
assert input_file != output_file, "Input and output file are the same."
assert input_file.exists(), "Input file does not exist."
assert not output_file.exists(), "Output file already exists. Remove it first."

# the keys to be added to each configuration in the cachefile
data_to_add = {
    "framework_time": 0,
    "strategy_time": 0,
    "benchmark_time": 0,
    "compile_time": 0,
    "verification_time": 0,
}
# the keys to be replaced in each configuration in the cachefile, values are evaluated
data_to_replace = {}
# the keys to be removed from each configuration in the cachefile, values do not matter
data_to_remove = {"times": "time"}


# read the input file
print("Retrieving data...")
with input_file.open(encoding="utf-8") as data_file:
    # get the relevant data out of the dictionary
    data: dict = json.load(data_file)
    assert isinstance(data, dict)
    cache = data["cache"]
    assert isinstance(cache, dict)

    # write the changes for each configuration
    print("Applying changes...")
    for key in cache.keys():
        config_results = cache[key]
        for dkey, dvalue in data_to_add.items():
            if dkey not in config_results:
                config_results[dkey] = dvalue
        for dkey in data_to_remove:
            if dkey in config_results:
                del config_results[dkey]
        cache[key] = config_results

    # write the changes to the file
    print("Writing to new file...")
    assert isinstance(cache, dict)
    data["cache"] = cache
    assert isinstance(data, dict)
    with output_file.open(mode="w+", encoding="utf-8") as result_file:
        json.dump(data, result_file, ensure_ascii=False, indent=4)
    print("  Done!")

# test whether output file can be read
print("Testing output file...")
assert output_file.exists()
with output_file.open(encoding="utf-8") as result_file:
    data: dict = json.load(result_file)
    assert isinstance(data, dict)
print("  Test successful")
