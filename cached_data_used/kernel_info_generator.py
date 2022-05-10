kernels_device_info = {
    'A100': {
        'name': 'A100',
        'displayname': 'A100',
        'kernels': {
            'GEMM': {
                'y_axis_upper_limit': 9.9,
            },
            'convolution': {
                'y_axis_upper_limit': 1.0,
            },
            'pnpoly': {
                'y_axis_upper_limit': 13.45,
            },
            'expdist': {
                'y_axis_upper_limit': 47,
            },
            'adding': {
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
                'absolute_difference': 2009.6310179999998,
                'mean': 59.99657231187346,
                'median': 44.194630999999994,
                'interquartile_range': 35.580132000000006,
                'std': 78.33888106834037,
                'y_axis_upper_limit': 22.8,
            },
            'convolution': {
                'absolute_optimum': 1.2208920046687126,
                'absolute_difference': 54.31783289462328,
                'mean': 5.419323945417355,
                'median': 3.4967679865658283,
                'interquartile_range': 2.537118246778846,
                'std': 6.105436583474108,
                'y_axis_upper_limit': 1.9,
            },
            'pnpoly': {
                'absolute_optimum': 12.325379967689514,
                'absolute_difference': 53.7067106962204,
                'mean': 21.491131074511443,
                'median': 17.629348874092102,
                'interquartile_range': 8.993487030267715,
                'std': 9.35671830095648,
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
                'absolute_difference': 3302.2904249999997,
                'mean': 100.36682823139897,
                'median': 70.92192349999999,
                'interquartile_range': 56.094469000000004,
                'std': 183.12100286273196,
                'y_axis_upper_limit': 37,
            },
            'convolution': {
                'absolute_optimum': 1.6253190003335476,
                'absolute_difference': 88.63640737906098,
                'mean': 11.22555501795815,
                'median': 4.9854245111346245,
                'interquartile_range': 5.56359495408833,
                'std': 16.096653621015548,
                'y_axis_upper_limit': 2.52,
            },
            'pnpoly': {
                'absolute_optimum': 26.968406021595,
                'absolute_difference': 97.67257422208786,
                'mean': 53.56028655904984,
                'median': 49.114556074142456,
                'interquartile_range': 21.304381847381592,
                'std': 17.732918013161168,
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
                'absolute_difference': 506.0608223482002,
                'mean': 101.68193528867094,
                'median': 65.9620063225122,
                'interquartile_range': 136.52849557432575,
                'std': 105.91338960852924,
                'y_axis_upper_limit': 4,
            },
            'Mishrasbird': {
                'absolute_optimum': 0.21930385091279447,
                'absolute_difference': 189.1096020845444,
                'mean': 114.03751804981042,
                'median': 116.30895572268682,
                'interquartile_range': 21.335740834611556,
                'std': 29.199523688431135,
                'y_axis_upper_limit': 51,
            },
            'Gomez-Levy': {
                'absolute_optimum': 0.0002861186645122249,
                'absolute_difference': 4.264675667668821,
                'mean': 1.4029071529901882,
                'median': 1.2672224221811628,
                'interquartile_range': 1.238731427823907,
                'std': 0.8450297778194089,
                'y_axis_upper_limit': 0.27,
            },
            'multimodal_sinewave': {
                'absolute_optimum': -1.9105822192681705,
                'absolute_difference': 3.6800166374910566,
                'mean': -0.17945949641637754,
                'median': -0.13695290145150701,
                'interquartile_range': 1.5534700018115082,
                'std': 0.9655453670304803,
                'y_axis_upper_limit': -1.7,
            },
            'predict_peak_mem': {
                "absolute_optimum": 6.392,
                "absolute_difference": 17.21,
                "mean": 12.282430393996249,
                "median": 11.7,
                "std": 2.907976801390474,
                "interquartile_range": 4.219000000000001
            }
        }
    }
}

if __name__ == "__main__":
    import json
    import numpy as np

    # gather all of the kernel information per device in a dictionary
    for device, device_dict in kernels_device_info.items():
        for kernel, kernel_dict in device_dict["kernels"].items():

            filestring = f"cache_files/{kernel}_{device}.json"
            try:
                with open(filestring, 'r') as fh:
                    print(f"Evaluating {filestring}")
                    orig_contents = fh.read()
                    try:
                        data = json.loads(orig_contents)
                    except json.decoder.JSONDecodeError:
                        contents = orig_contents[:-1] + "}\n}"
                        try:
                            data = json.loads(contents)
                        except json.decoder.JSONDecodeError:
                            contents = orig_contents[:-2] + "}\n}"
                            data = json.loads(contents)
                    cache = data['cache']
                    keys, values = list(cache.keys()), cache.values()
                    times = np.array(list(v['time'] for v in values))
                    times = times[times < 1e20]

                    # gather statistics
                    sorted_times = np.sort(times).tolist()
                    minimum = sorted_times[0]
                    maximum = sorted_times[-1]
                    mean = np.mean(times)
                    median = np.median(times)
                    std = np.std(times)
                    q75, q25 = np.percentile(times, [75, 25])
                    iqr = q75 - q25
                    # print(
                    #     f"'absolute_optimum': {minimum},\n'absolute_difference': {maximum-minimum},\n'mean': {mean},\n'median': {median},\n'interquartile_range': {iqr},\n'std': {std},"
                    # )

                    # write to the dict
                    to_add = {
                        'absolute_optimum': minimum,
                        'absolute_difference': maximum - minimum,
                        'mean': mean,
                        'median': median,
                        'std': std,
                        'interquartile_range': iqr,
                        'sorted_times': sorted_times,
                        'size': len(times),
                    }
                    for key, value in to_add.items():
                        kernel_dict[key] = value

            except FileNotFoundError:
                print(f"Couldn't find {filestring}")
            print("")

    # write the dictionary to file
    # print(kernels_device_info)
    kernels_device_info_data = json.dumps(kernels_device_info)

    # Write the string to a file
    with open("kernel_info.json", 'w') as file:
        file.write(kernels_device_info_data)
