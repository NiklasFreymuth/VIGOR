REWARD_PRIOR_NAME = "q0"
PHI_NAME = "phi"
REGRESSIVE_REWARD = "regression_reward"
EIM_DISCRIMINATOR = "eim_dre"
NORMALIZATION_PARAMETER_FILE = "normalization_parameters"
NETWORK_KWARGS_FILE = "network_kwargs"
POLICY_NAME = "policy"
REWARD_FOLDER = "reward_parameters"
POLICY_FOLDER = "policy_parameters"
SAVE_DIRECTORY = "checkpoints"


def format_iteration(iteration):
    return "{:04d}".format(iteration)
