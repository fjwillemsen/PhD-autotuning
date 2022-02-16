import sys
from typing import Tuple


def bootstrap_hyperparamtuning_kernel(invalid_flag, **kwargs) -> Tuple[list, dict]:
    # The imports here are necessary for calling a python kernel directly, because the global imports are not seen.
    # Modules are cached in sys.modules, so duplicate imports should not cause significant overhead.
    from importlib import import_module
    import numpy as np
    import warnings
    import json
    # from gpytorch.utils.errors import NanError

    # read the kernel info dictionary from file
    with open("kernel_info.json") as file:
        kernels_device_info_data = file.read()
    kernels_device_info = json.loads(kernels_device_info_data)

    # strategy = "bayes_opt"
    strategy = "bayes_opt_GPyTorch_lean"

    # device_names = list()
    # kernel_names = ['Rosenbrock', 'Mishrasbird', 'Gomez-Levy']
    # device_name = 'generator'

    # kernel_names = ['convolution']
    kernel_names = ['convolution', 'pnpoly', 'GEMM']
    device_name = 'RTX_2070_SUPER'

    kernels = list(import_module(kernel_name) for kernel_name in kernel_names)
    kernel_device_info = kernels_device_info[device_name]['kernels']

    invalid_value = 1e20
    invalid_return_value = [invalid_value] * len(kernels)

    warncategories = [
        'AvoidedLossSurgeWarning', 'NumericalWarning', 'NotPSDTrainingWarning', 'NaNTrainingWarning', 'NotPSDPredictionWarning', 'NaNPredictionWarning',
        'ResetModelWarning', 'MultipleMinimaWarning', 'AlreadyEvaluatedConflict'
    ]
    warn_counter = [0 for _ in warncategories]

    def return_invalid(err_reason: str):
        print("Return invalid")
        if not bool(invalid_flag.value):
            invalid_flag.value = int(True)
            print("Invalid flag raised", flush=True)
        warning_counter = dict(zip(warncategories, warn_counter))
        warning_counter['invalid_reason'] = err_reason
        return invalid_return_value, warning_counter

    mean_normalized_errors = list()
    mean_weighted_positions = list()
    for kernel_index, kernel in enumerate(kernels):
        # first check if the invalid flag has been raised by another process, so we avoid running each kernel before we find out it wasn't necessary
        if bool(invalid_flag.value):
            return invalid_return_value

        # execute the configuration while catching warnings
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("ignore", category=UserWarning)
            # warnings.filterwarnings("ignore", category=UserWarning)
            results, env = kernel.tune(device_name, strategy, strategy_options=kwargs, verbose=False, quiet=True, simulation_mode=True)
            # warnings.filterwarnings("default", category=UserWarning)
            for warn in warns:
                if warn.category.__name__ in warncategories:
                    warn_counter[warncategories.index(warn.category.__name__)] += 1
                else:
                    print("Not catched: ", warn)
        # except (FloatingPointError, NanError, ValueError):
        #     return return_invalid()

        # check if there are only invalid values
        if min(results, key=lambda x: x['time'])['time'] >= invalid_value:
            return return_invalid(err_reason="No valid value found")

        # # use the kernel statistics to obtain the Mean Normalized Error
        # kernel_name = kernel_names[kernel_index]
        # absolute_optimum = kernel_device_info[kernel_name]['absolute_optimum']
        # median = kernel_device_info[kernel_name]['median']
        # normalized_errors = list()
        # split_results = range(40, len(results) + 1, 20)
        # for count, index in enumerate(split_results):
        #     selected_res = results[19:index - 1]
        #     f = min(selected_res, key=lambda x: x['time'])['time']
        #     if f < invalid_value:
        #         # multiply by count to make sure that later iterations count more heavily, optionally use sqrt(count)
        #         # TODO match the kernels to a distribution, use a CDF to see how likely the obtained value or better is, use that as a measure of how good it is
        #         normalized_errors.append(np.sqrt(count) * ((f - absolute_optimum) * 1))
        # if len(normalized_errors) <= 0:
        #     return return_invalid(err_reason="No valid normalized value found")
        # # divide by the length so invalids are punished
        # mean_normalized_errors.append(np.sum(normalized_errors) / len(split_results))

        # use the kernel statistics to obtain the Mean Position
        kernel_name = kernel_names[kernel_index]
        cache_size: int = kernel_device_info[kernel_name]['size']
        cache_sorted_times: list = kernel_device_info[kernel_name]['sorted_times']
        positions = list()
        split_results = range(40, len(results) + 1, 20)
        for count, index in enumerate(split_results):
            selected_res = results[19:index - 1]
            f = min(selected_res, key=lambda x: x['time'])['time']
            if f < invalid_value:
                # multiply by count to make sure that later iterations count more heavily, optionally use sqrt(count)
                # TODO match the kernels to a distribution, use a CDF to see how likely the obtained value or better is, use that as a measure of how good it is
                cache_position = cache_sorted_times.index(f)
                positions.append(count * (cache_position / cache_size))
            else:
                print("Invalid")
        if len(positions) <= 0:
            return return_invalid(err_reason="No valid normalized value found")
        # divide by the length so invalids are punished
        mean_weighted_positions.append(np.sum(positions) / len(split_results))

    # return the MNE values per kernel and warnings
    warning_counter = dict(zip(warncategories, warn_counter))
    return mean_weighted_positions, warning_counter

    # return the MRE value
    # mean_MRE = np.mean(MREs)
    # if np.isnan(mean_MRE) or mean_MRE == np.nan:
    #     return invalid_value
    # return mean_MRE

    # calculate the Mean Deviation Factor (MDF) per kernel
    MDFs = list()
    for mean_MAE in mean_MAEs:
        MDFs.append(mean_MAE / grandmean_MAE)
    print(f"MDFs: {MDFs}")
    print(f"grand mean MAE: {grandmean_MAE}")


if __name__ == "__main__":
    # print(sys.argv)
    kwdict = dict(arg.split('=') for arg in sys.argv[1:])
    print(
        f"result_value={bootstrap_hyperparamtuning_kernel(**kwdict)}")    # do not touch, this is read by the Python subprocess device handler in Kernel Tuner!
