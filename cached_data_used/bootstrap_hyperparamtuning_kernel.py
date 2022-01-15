from importlib import import_module
import numpy as np
from time import perf_counter

kernels_device_info = {
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
    'RTX_2070_SUPER': {
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
            'multimodal_sinewave': {
                'absolute_optimum': -1.92,
                'y_axis_upper_limit': -1.7,
            }
        }
    }
}

def bootstrap_hyperparamtuning_kernel(strategy="bayes_opt_GPyTorch_lean", iterations = 14, **kwargs):
    device_names = list()
    kernel_names = ['Rosenbrock', 'Mishrasbird', 'Gomez-Levy']
    kernels = list(import_module(kernel_name) for kernel_name in kernel_names)
    device_name = 'generator'
    kernel_device_info = kernels_device_info[device_name]['kernels']

    tune_options = {
        'verbose': False,
        'quiet': True,
        'simulation_mode': True
    }

    timing_dict = {}
    mean_MAEs = list()
    for kernel_index, kernel in enumerate(kernels):
        kernel_name = kernel_names[kernel_index]
        absolute_optimum = kernel_device_info[kernel_names[kernel_index]]['absolute_optimum']
        timing_dict[kernel_name] = list()
        MAEs = list()
        for _ in range(iterations):
            start_time = perf_counter()
            results, env = kernel.tune(device_name, strategy, strategy_options=kwargs, verbose=False, quiet=True, simulation_mode=True)
            timing_dict[kernel_name].append(perf_counter() - start_time)
            MAE = 0
            fplus = absolute_optimum
            split_results = range(40, 220+1, 20)
            for index in split_results:
                selected_res = results[20:index]
                f = min(selected_res, key=lambda x:x['time'])['time']
                MAE += abs(f - fplus)
            MAE = MAE / len(split_results)
            MAEs.append(MAE)
        mean_MAEs.append(np.mean(MAEs))

    grandmean_MAE = np.mean(mean_MAEs)
    return grandmean_MAE

    # calculate the Mean Deviation Factor (MDF) per kernel
    MDFs = list()
    for mean_MAE in mean_MAEs:
        MDFs.append(mean_MAE / grandmean_MAE)
    print(f"MDFs: {MDFs}")
    print(f"grand mean MAE: {grandmean_MAE}")
