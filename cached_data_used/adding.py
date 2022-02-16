import sys
from .adding_kernel.adding_kernel_wrapper import tune as wrapped_tune


def tune(device_name, strategy="bayes_opt_GPyTorch_lean", strategy_options=None, verbose=True, quiet=False, simulation_mode=True):
    return wrapped_tune(device_name, strategy, strategy_options, verbose, quiet, simulation_mode)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./adding.py [device name]")
        exit(1)

    device_name = sys.argv[1]

    tune(device_name)
