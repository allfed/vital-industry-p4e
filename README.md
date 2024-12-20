# Vital Industry P4E Calculations

![Testing](https://github.com/allfed/vital-industry-p4e/actions/workflows/testing.yml/badge.svg?branch=main)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

- Supplementary code for the [Guesstimate model](https://www.getguesstimate.com/models/23914) used in the paper "Developing back-up plans to protect vital sectors in super-pandemics" (preprint link coming soon).
- Contains calculations for the probability that all rapidly-accessible pandemic-proof PPE (P4E) can cover vital worker requirements and code for generating Figure 1 from the paper.

## Installation
To install this repo, we recommend setting up a virtual environment. This will ensure that the package and its dependencies are isolated from other projects on your machine, which can prevent conflicts and make it easier to manage your dependencies. Here are the steps to follow:

* Create a virtual environment using either conda or mamba by running the command `conda env create -f environment.yml`. This will create an environment called "vital-industry-p4e". A virtual environment is like a separate Python environment, which you can think of as a separate "room" for your project to live in, it's own space which is isolated from the rest of the system, and it will have it's own set of packages and dependencies, that way you can work on different projects with different versions of packages without interfering with each other.

* Activate the environment by running `conda activate vital-industry-p4e`. This command will make the virtual environment you just created the active one, so that when you run any python command or install any package, it will do it within the environment.

* Install the package by running `pip install -e .` in the main folder of the repository. This command will install the package you are currently in as a editable package, so that when you make changes to the package, you don't have to reinstall it again.

If you encounter any issues, feel free to open an issue in the repository.
