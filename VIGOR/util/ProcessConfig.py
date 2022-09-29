import copy

from util import Defaults as d
from util.Types import *

def process_config(current_config: Dict[Key, Any], full_config: Optional[Dict[Key, Any]] = None) -> Dict[Key, Any]:
    """
    Recursively parses a given dictionary by going through its keys and adapting the values into a suitable and most
    importantly standartized format
    Also removes subdictionaries that are not important to the current task
    Args:
        current_config: The current (sub-)config
        full_config: The full config for reference. Used for filtering out subconfigs that are not needed for the
        specific run

    Returns: The parsed dictionary

    """
    if full_config is None:
        full_config = copy.deepcopy(current_config)
    parsed_config = {}
    for key, value in sorted(current_config.items(), key=lambda x: x[0].startswith(d.LOG_PREFIX)):
        # assure that logs happen last
        if isinstance(value, dict):
            parsed_config[key] = process_config(current_config=value, full_config=full_config)
        elif key.startswith(d.LOG_PREFIX):
            if isinstance(value, int) and value > 0:
                parsed_value = int(2 ** value)
            elif isinstance(value, int) and value < -30:  # round very small values to 0
                parsed_value = 0
            else:
                parsed_value = 2 ** value

            parsed_config[key.replace(d.LOG_PREFIX, "", 1)] = parsed_value
        elif key == "dropout":
            parsed_config[key] = value if 0 < value < 1 else False
        elif isinstance(value, float) and value.is_integer():
            parsed_config[key] = int(value)  # parse to int when possible
        else:
            parsed_config[key] = value
    return parsed_config
