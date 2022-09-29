import tqdm


class TrainerProgressBar:
    """
    Utility class for a custom tqdm progress bar
    """

    def __init__(self, num_epochs: int, verbose: bool, position: int = 0):
        self._verbose = verbose
        if verbose:
            scalar_tracker_format = '{desc}'
            self._scalar_tracker = tqdm.tqdm(total=num_epochs, desc="Scalars", bar_format=scalar_tracker_format,
                                             position=position,
                                             leave=True)
            progress_bar_format = '{desc} {n_fmt:' + str(
                len(str(num_epochs))) + '}/{total_fmt}|{bar}|{elapsed}<{remaining}'
            self._progress_bar = tqdm.tqdm(total=num_epochs, desc='Epoch', bar_format=progress_bar_format,
                                           position=position + 1,
                                           leave=True)
        else:
            self._scalar_tracker = None
            self._progress_bar = None

    def close(self):
        if self._progress_bar is not None:
            self._progress_bar.close()
            self._progress_bar = None
        if self._scalar_tracker is not None:
            self._scalar_tracker.close()
            self._scalar_tracker = None

    def update(self, epoch_scalars: dict):
        if self._verbose:  # progress bar
            formatted_epoch_scalars = {k: "{:.2e}".format(v) for k, v in epoch_scalars.items()}
            description = ("Scalars: " + "".join(
                [str(k) + "=" + v + ", " for k, v in formatted_epoch_scalars.items()]))[:-2]
            self._scalar_tracker.set_description(description)
            self._progress_bar.update(1)
