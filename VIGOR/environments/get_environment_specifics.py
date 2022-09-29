from util.Types import *


def get_environment_specifics(task_name: str, config: ConfigDict, data_dict: ValueDict) -> ValueDict:
    """
    Build a specific environment of the given task from the config and the information provided in the data_dict
    Args:
        task_name:
        config: Configuration file for this run
        data_dict: Contains information about data to use for the task.
            Most importantly, has a list of train/validation/test contexts to use

    Returns:

    """
    if task_name == "planar_reacher":
        from environments.planar_reacher.planar_reacher_specifics \
            import get_planar_reacher_specifics
        environment_specifics = get_planar_reacher_specifics(config=config, data_dict=data_dict)
    elif task_name == "panda_reacher":
        from environments.panda_reacher.panda_reacher_specifics \
            import get_panda_reacher_specifics
        environment_specifics = get_panda_reacher_specifics(config=config, data_dict=data_dict)
    elif task_name == "box_pusher":
        from environments.box_pusher.box_pusher_specifics \
            import get_box_pusher_specifics
        environment_specifics = get_box_pusher_specifics(config=config, data_dict=data_dict)
    elif task_name == "table_tennis":
        from environments.table_tennis.table_tennis_specifics \
            import get_table_tennis_specifics
        environment_specifics = get_table_tennis_specifics(config=config, data_dict=data_dict)
    else:
        raise ValueError("Task '{}' does not exist".format(task_name))
    return environment_specifics
