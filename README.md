# Inferring Versatile Behavior from Demonstrations by Matching Geometric Descriptors

## Install
We highly recommend using a [conda environment](https://docs.conda.io/en/latest/) before installing 
the requirements for this repository.

Requirements can be installed with 
```pip install -r requirements.txt```

## Code Structure
The Code is written in pure Python. The general setup is as follows:

- "Baselines" contains state-action based environments and different imitation learning baselines for them
- "VIGOR" contains VIGOR and other trajectory based baselines.
  - All experiments use [cw2](https://www.github.com/ALRhub/cw2.git) for distributing and recording purposes. 
    - An example _without_ cw2 is given in 'VIGOR/quickstart.py'.

  - The `configs` directory contains `.yaml` files that specify individual experiments. , in which individual sections describe the parameters for a given cw2 run. The ```default.yaml```file contains a short description of each parameter. A run can be called with given parameters using the command ```python main.py configs/config_file.yaml -e run_name -o```
    - To run an experiment on a slurm-based cluster, simply append a ```-s``` to the above, i.e., ```python main.py configs/configs.yaml -e experiment -o -s --nocodecopy```
    - We provide an overview of the experiments of the paper in the `experiments.sh` file. Note that the experiments contain multiple seeds, so they might take a while to run. You can adapt the corresponding `.yaml` files accordingly, e.g., to run a single seed.

  - Experiments will be automatically recorded and logged in "/VIGOR/experiments". Options for this can be specified via the parameters. Recordings include visualizations for the given tasks, model and reward parameters and graphs for different performance metrics over time. The used config as well as the logged statements are also recorded for convenience
  - Additionally, all metrics can be tracked with WandB
  
## Requirements Python 
Tested with Python 3.8.12. on Windows and Ubuntu. A list of all required packages, including version numbers can be found in req.txt and can
be installed with
