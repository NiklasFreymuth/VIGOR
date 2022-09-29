from functools import reduce
from util.Types import *
from util.Functions import get_from_nested_dict
import numpy as np
import torch
import copy
from torch import optim
from torch.utils.data import DataLoader
import util.Defaults as d
from util.pytorch.architectures.Network import Network
from util.pytorch.architectures.SharedSequenceDiscriminator import SharedSequenceDiscriminator
from util.pytorch.architectures.ConvolutionalSequenceDiscriminator import ConvolutionalSequenceDiscriminator
from util.pytorch.architectures.LSTMDiscriminator import LSTMDiscriminator
from util.pytorch.EarlyStopping import initialize_early_stopping
from util.pytorch.training.TrainerProgressBar import TrainerProgressBar
from util.pytorch.training.TrainingUtil import format_history, compute_loss_and_metrics


class NetworkTrainer:
    def __init__(self, network: Network, network_config: Dict[str, Any],
                 primary_loss_function: LossFunction):
        """
        Initializes a trainer for a given neural network.
        Args:
            network: a pytorch network
            network_config: Contains information about how to train a network.
              Includes batch_size, number of epochs to train, early_stopping and more.
            primary_loss_function: A loss function f(samples, targets, weights)-> loss
        """
        if network_config.get("use_gpu"):  # Whether to use a GPU for training or not
            self._gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._gpu_device = None

        self._learning_rate: float = network_config.get("learning_rate")
        self._l2_norm = network_config.get("regularization").get("l2_norm")

        self._network = network
        self._optimizer = optim.Adam(self.network.parameters(), lr=self._learning_rate, weight_decay=self._l2_norm)

        self._batch_size = network_config.get("batch_size")
        self._epochs = network_config.get("epochs")
        self._verbose = network_config.get("verbose")

        self._primary_loss_function = primary_loss_function

        if isinstance(self.network, (SharedSequenceDiscriminator,
                                     LSTMDiscriminator,
                                     ConvolutionalSequenceDiscriminator)):
            self._stepwise_loss = get_from_nested_dict(dictionary=network_config,
                                                       list_of_keys=["time_series", "stepwise_loss"],
                                                       raise_error=True)
        else:
            self._stepwise_loss = False

        self._early_stopping = initialize_early_stopping(early_stopping_config=network_config.get("early_stopping", {}))

    def __call__(self, train_data_loader: DataLoader, validation_data_loader: DataLoader = None):
        """
        Wrapper for .fit().
        Args:
            train_data_loader: DataLoader for training. Must refer to a DictionaryDataset
            validation_data_loader: DataLoader for validation. Must refer to a DictionaryDataset
        Returns: A history of the fit, including the loss value for each epoch as well as potential other metrics
        like val_loss, acc etc.


        """
        return self.fit(train_data_loader=train_data_loader,
                        validation_data_loader=validation_data_loader)

    def fit(self, train_data_loader: DataLoader, validation_data_loader: DataLoader = None):
        """
        Fits the network to the data provided in train_data_loader and evaluates it using evaluation_data_loader
        Args:
            train_data_loader: DataLoader for training. Must refer to a DictionaryDataset
            validation_data_loader: DataLoader for validation. Must refer to a DictionaryDataset
        Returns: A history of the fit, including the loss value for each epoch as well as potential other metrics
        like val_loss, acc etc.

        """
        if self._early_stopping is not None:
            self._early_stopping.reinitialize()
        progress_bar = TrainerProgressBar(num_epochs=self._epochs, verbose=self._verbose)

        history = []

        for epoch in range(self._epochs):
            epoch_scalars = self._run_epoch(dict_data_loader=train_data_loader, train_network=True)
            if validation_data_loader is not None:
                epoch_val_scalars = self._run_epoch(dict_data_loader=validation_data_loader, train_network=False)
                epoch_scalars = {**epoch_scalars, **{d.VALIDATION_PREFIX + k: v for k, v in epoch_val_scalars.items()}}

                if self._early_stopping is not None:
                    stop = self._early_stopping(epoch_scalars[d.VALIDATION_PREFIX + d.TOTAL_LOSS],
                                                network=self._network)
                    if stop:
                        progress_bar.close()
                        if self._verbose:
                            print(
                                "Early stopping due to no improvement in {} epochs".format(
                                    self._early_stopping.patience))
                        break
            history.append(epoch_scalars)

            progress_bar.update(epoch_scalars=epoch_scalars)

        progress_bar.close()
        return format_history(history)

    def _run_epoch(self, dict_data_loader: DataLoader, train_network: bool = False) -> dict:
        """
        Wrapper for running a whole dataloader through the network once. If train_network, the data is used to update
         the model. If not, then the weights are not updated. In both cases the average loss over all data is returned.
        Args:
            dict_data_loader: A torch DataLoader using a DictionaryDataset
            train_network: Whether to update the parameters or not

        Returns: A dict scalars containing the loss and potentially other scalars depending on the kind of training
        """

        def __run():  # wrapper for the "with" thing as a nullcontext is Python 3.7+ :(
            return zip(*[self._run_batch(batch, train_network=train_network) for batch in
                         dict_data_loader])

        if train_network:  # tell the network that it is about to be trained. Important for dropout etc.
            self._network.train()
            batch_samples, scalars = __run()
        else:  # now its about to be evaluated. Again changes dropout etc.
            self._network.eval()
            with torch.no_grad():
                batch_samples, scalars = __run()

        weighted_scalars = [{k: num_samples * v
                             for k, v in batch_scalars.items()}
                            for num_samples, batch_scalars in
                            zip(batch_samples, scalars)]
        total_samples = np.sum(batch_samples)
        reduced_scalars = reduce(
            lambda a, b: {k: a[k] + b[k] for k in a},
            weighted_scalars,
            dict.fromkeys(weighted_scalars[0], 0.0))
        normalized_scalars = {k: (v / total_samples) for k, v in reduced_scalars.items()}
        return normalized_scalars

    def _to_potential_gpu(self, tensor: Union[None, torch.Tensor]) -> Union[None, torch.Tensor]:
        if tensor is not None:
            if self._gpu_device and not tensor.is_cuda:
                return tensor.to(device=self._gpu_device)
            else:
                return tensor
        else:
            return None

    def _run_batch(self, tensors: dict, train_network: bool = False) -> Tuple[int, dict]:
        """
        Runs and evaluates the network for the given samples and targets. If train_network, then the network is updated
        based on the results
        Args:
            tensors: A dictionary of named tensors. Always contains "samples" and "targets" as well as potentially
            more tensors for different kinds of training
            train_network: Whether to update the parameters or not
        Returns: A dictionary of scalars

        """

        samples: torch.Tensor = self._to_potential_gpu(tensors.get(d.SAMPLES, None))
        targets = self._to_potential_gpu(tensors.get(d.TARGETS, None))
        weights = self._to_potential_gpu(tensors.get(d.WEIGHTS, None))

        if weights is not None:
            weights = weights / torch.mean(weights)  # since validation and train weights may otherwise be slightly off
        batch_size: int = len(samples)

        output_dict: dict = self._network(samples, as_dict=True)

        scalars = {d.TOTAL_LOSS: 0}
        loss = self._to_potential_gpu(torch.zeros(1))

        loss, scalars = compute_loss_and_metrics(batch_size=batch_size, loss=loss, output_dict=output_dict,
                                                 scalars=scalars, targets=targets, weights=weights,
                                                 loss_function=self._primary_loss_function,
                                                 stepwise_loss=self._stepwise_loss
                                                 )

        if train_network:  # optimize for a step
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        scalars[d.TOTAL_LOSS] = loss.item()

        return batch_size, scalars  # number of predictions and different metrics and losses

    @property
    def network(self) -> Network:
        return self._network

    @network.setter
    def network(self, new_network: Network):
        """
        Sets a new network by copying the given network and creating a new optimizer for it
        Args:
            new_network:

        Returns:

        """
        self._network = copy.deepcopy(new_network)
        self._optimizer = optim.Adam(self.network.parameters(), lr=self._learning_rate, weight_decay=self._l2_norm)
