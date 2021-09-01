# Bayesian Optimization for auto-tuning GPU kernels
Encompassing repository for software written or used, as well as data generated or used in the context of the prospecting paper "Bayesian Optimization for auto-tuning GPU kernels", specifically to comply with the Artifact Description and Artifact Evaluation under the SuperComputing 2021 reproducability initiative. 

_This repository is Apache 2.0 licensed._

## How to use
1. Clone this repository
  This repository contains [submodules](https://git-scm.com/book/nl/v2/Git-Tools-Submodules). 
  To register a submodule, move into the directory and use `git submodule init`. 
  Next, execute `git submodule update`. 

2. Create a Python virtual environment
  - Initialize a Python virtual environment using `virtualenv <name>`.
  - Activate the created virtual environment.
  - Use `pip install -r venv_requirements.txt` to install the required packages in the virtual environment. 
  When packages are changed (added, deleted, updated), use `pip freeze > venv_requirements.txt` to make these changes accessible.
  
 3. Get to know Kernel Tuner **_[optional]_**
  - Initialize the [Kernel Tuner repository](https://github.com/benvanwerkhoven/kernel_tuner/tree/b54b1168532a22a25ec286eb72073bf7a424f3d3) and read the Read Me.
  - If interested, follow the [Kernel Tuner Tutorial](https://github.com/benvanwerkhoven/kernel_tuner_tutorial/tree/b9a7dbba44cbf31673436411dced85383aa44dac).

## How to reproduce results
1. Move into the `experiments` directory.
2. Running the visualization script is simple, e.g.: `python experiments.py -kernels GEMM convolution pnpoly`. The `-kernels` keyword denotes the list of kernel names to visualize (space-seperated), while the `-devices` keyword denotes a list of GPUs to visualize (space-seperated, by default `GTX_TITAN_X`). Running this script will result in Mathplotlib plots (by default resulting in the plots of figure 1 in the paper) and output in the terminal. 
3. The `experiments.py` script takes care of running each specified algorithm in the simulation mode of Kernel Tuner with the specified options, then aggregates the resulting data in a cache file in the folder `cached_data_used/cached_experiments`. On each run, it first checks whether a cached version is available in this folder, as running each specified algorithm for the specifed number of times (`repeats`) can take a while. Cached versions of the runs used in the paper are supplied in this repository. To recalculate the results, delete the files in `cached_data_used/cached_experiments`, and run step 2 again (this might take a while depending on the options!). 
4. If necessary, edit the code in the `main` body (starting at line 1471) of `experiments.py`. Different types of plots and other options are available here. Additionally, if necessary, change the `self.strategies` variable at line 426 to point to a dictionary of choice containing the strategies to compare. An exhaustive, descripted list of the dictionaries for the strategies comparisons used in the prospective paper is found in the file `experiments_strategies.py`, also in the `experiments` folder. 
