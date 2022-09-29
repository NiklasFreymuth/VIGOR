RECORDING_DIR = "experiments"
NETWORK_VISUALIZATION = "network_schematic"
ENVIRONMENTS_FOLDER = "environments"

# training, metrics and losses
ACCURACY = "acc"
TOTAL_LOSS = "loss"
BINARY_CROSS_ENTROPY = "bce"
MEAN_SQUARED_ERROR = "mse"
COMPARISON_BASED_LOSS = "drex"
ROOT_MEAN_SQUARED_ERROR = "rmse"
MEAN_ABSOLUTE_ERROR = "mae"
VALIDATION_PREFIX = "val_"
STEPWISE_L1_LOSS = "sl1"
STEPWISE_L2_LOSS = "sl2"


# train/test/validation
TRAIN = "train"
TEST = "test"
VALIDATION = "validation"
DREX_TRAIN = "drex_train"

# Plots
Y_AXIS_TICKS = 5

# Files
FINAL_METRIC_NAME = "final_metrics"
METRIC_FILE_NAME = "metrics"
MULTI_METRIC_FILE_NAME = "multi_metrics"
NETWORK_HISTORY_PLOT_NAME = "network_history"
TASK_VISUALIZATION_DIRECTORY = "Vis"
ADDITIONAL_PLOT_DIR = "additional_plots"

# Config
LOG_PREFIX = "log_"


# Different Network outputs
PREDICTIONS = "predictions"
STEPWISE_LOGITS = "stepwise_logits"


# Training data
SAMPLES = "samples"
TARGETS = "targets"
WEIGHTS = "weights"

# shared
VELOCITY_PENALTY = "velocity_penalty"
ACCELERATION_PENALTY = "acceleration_penalty"

# reaching task
DISTANCE_TO_CENTER = "distance_to_center"
DISTANCE_TO_BOUNDARY = "distance_to_boundary"
SUCCESS = "success_rate"

# time-series task log_densities/reward-decompositions
OBSTACLE_COLLISION_PENALTY = "obstacle_collision_penalty"
ACCELERATION_LOG_DENSITY = "acceleration_log_density"
VELOCITY_LOG_DENSITY = "velocity_log_density"
TARGET_LOG_DENSITY = "target_log_density"
TOTAL_REWARD = "total_reward"

# box pusher
TARGET_DISTANCE_PENALTY = "target_distance_penalty"
TARGET_IMPROVEMENT = "target_improvement"
