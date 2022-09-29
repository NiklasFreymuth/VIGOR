import torch
from torch.utils.data import Dataset


class DictionaryDataset(Dataset):
    def __init__(self, **kwargs):
        if "samples" in kwargs:
            self.num_samples = len(kwargs.get("samples"))
        elif "targets" in kwargs:
            self.num_samples = len(kwargs.get("targets"))
        else:
            raise ValueError("Need to provide either samples or targets")

        self.dictionary_dataset = {}
        for name, tensor in kwargs.items():
            if tensor is not None:
                assert self.num_samples == len(tensor), "All tensors must have the same length".format(tensor)
                self.dictionary_dataset[name] = tensor

    def __getitem__(self, index) -> dict:
        """
        Receives the items of all initialized dictionary_dataset at the given list of indices
        Args:
            index:

        Returns:

        """
        return {k: v[index] for k, v in self.dictionary_dataset.items()}

    def __len__(self) -> int:
        return self.num_samples
