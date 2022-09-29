import logging
import os
import sys


class CustomFormatter(logging.Formatter):
    """
    Custom Formatter to allow for uniform and pretty console prints
    """

    def __init__(self):
        # prepare a standard logging via [name of the logger] logged message
        super().__init__()
        self.std_formatter = logging.Formatter('[%(name)s] %(message)s')
        self.red_formatter = logging.Formatter('[%(asctime)s] %(message)s')

    def format(self, record: logging.LogRecord):
        if record.levelno <= logging.ERROR:
            return self.std_formatter.format(record)  # everything but CRITICAL is formatted via standard
        else:
            return self.red_formatter.format(record)  # CRITICAL logs are formatted via red


def get_logger(name: str, path: str = None) -> logging.Logger:
    """
    Sets the logging configs to pipe logs to a recording.log file
    :param path:
    :return: the file_handler that deals with the logging
    """
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    formatter = CustomFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)  # log everything but DEBUG logs
    logger.addHandler(stream_handler)

    if path is not None:  # add filehandler that writes to "recording.log" in the given path
        file_handler = logging.FileHandler(os.path.join(path, "recording.log"))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    return logger


def remove_handlers(logger):
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    return logger


def log_heading(logger, string_to_frame: str, characters_per_side: int = 4, space: bool = True, character: str = "-",
                upper: bool = True):
    """
    Utility function to frame a string in a given character for nicer printing
    Args:
        string_to_frame:
        characters_per_side:
        space:
        character:
        upper:

    Returns:

    """
    total_length = (2 * (characters_per_side + space) + len(string_to_frame))
    if upper:
        string_to_frame = string_to_frame.upper()
    if space:
        string_to_frame = " " + string_to_frame + " "
    logger.info(total_length * character)
    logger.info(characters_per_side * character + string_to_frame + characters_per_side * character)
    logger.info(total_length * character)
