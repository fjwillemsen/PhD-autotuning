# Master Thesis
Encompassing repository for software written during the master project at the University of Amsterdam &amp; Free University Amsterdam

## How to use
1. Clone this repository
  This repository contains [submodules](https://git-scm.com/book/nl/v2/Git-Tools-Submodules). 
  To register a submodule, `cd` into the directory and use `git submodule init`. 
  Next, execute `git submodule update`. 

2. Create a Python virtual environment
  - Using `virtualenv <name>`
  - Activate the virtual environment
  - Use `pip install -r thesis_venv_requirements.txt` to install the required packages in the virtual environment. 
  When packages are changed (added, deleted, updated), use `pip freeze > thesis_venv_requirements.txt` to make these changes accessible.
