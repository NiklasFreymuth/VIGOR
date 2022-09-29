import numpy as np
from util.pytorch.architectures.Network import Network
from torch.utils.data import DataLoader
from util.pytorch.training.NetworkTrainer import NetworkTrainer
from util.Functions import interweave
import util.Defaults as d
from util.Types import *
from util.pytorch.training.TrainingUtil import dataloader_from_dictionary_data
from util.pytorch.NetworkBuilder import NetworkBuilder

"""
Wrapper class for training and handling training data when fitting the EIM network/log density ratio estimator
All parameters, data, weights and (side-)models relevant for training can be given to this class via the appropriate 
variables.
Calling the .prepare_data_and_fit(network) subroutine will use all these to train the given model. 
The model will be returned.
"""


class VIGORTrainer(NetworkTrainer):
    def __init__(self, config: ConfigDict, network_type: str, expert_observations: np.ndarray):
        """
        Initialize the training class by giving it all relevant hyperparameters.
        
        """
        network_config = config.get("network")

        network_builder = NetworkBuilder(input_shape=expert_observations.shape[1:],
                                         network_config=network_config,
                                         network_type=network_type,
                                         loss="bce")
        network, discriminator_loss_function = network_builder()

        self._validation_split = network_config.get("validation_split", 0)

        self._expert_observations = expert_observations.astype(np.float32)
        self._learner_observations = None

        super().__init__(network=network, network_config=network_config,
                         primary_loss_function=discriminator_loss_function)

    @property
    def learner_observations(self) -> np.ndarray:
        return self._learner_observations

    @learner_observations.setter
    def learner_observations(self, samples: np.ndarray):
        self._learner_observations = samples.astype(np.float32)

    def prepare_data_and_fit(self, reset_learner_observations=True) -> Dict:
        """
        Prepares the data by creating DataLoaders and then subsequently calls the fit function.
        Will fit the model in-place and return the history of the fit.
        Args:
            reset_learner_observations: If True, resets the observations of the learner after fitting.
            This forces the algorithm to set new data before fitting again, thus making things less prone for error

        Returns: A history of the fit as a dictionary {metric: [value over epochs]}

        """
        assert self._learner_observations is not None, "Must have learner observations for training"
        np.random.shuffle(self._learner_observations)
        train_data_loader, validation_data_loader = self._get_dict_data_loaders()
        history = self.fit(train_data_loader=train_data_loader, validation_data_loader=validation_data_loader)
        if reset_learner_observations:
            self._learner_observations = None
        return history

    def _get_dict_data_loaders(self) -> Tuple[DataLoader, Union[DataLoader, None]]:
        """
        Creates a training data loader that is responsible for partitioning the training data into batches.
        May also create a validation data loader if self._validation_split > 0
        Creates targets and weights for the data and interweaves it in a way that allows for a 
        consistent validation split.
        :return: A tuple (expert_train_data_loader, learner_train_data_loader, expert_validation_data_loader,
            learner_validation_data_loader) where the validation_data_loaders may be None
        """
        data = self._get_data()  # dictionary of all data
        data_loaders = dataloader_from_dictionary_data(data,
                                                       batch_size=self._batch_size,
                                                       validation_split=self._validation_split)
        train_data_loader, validation_data_loader = data_loaders

        return train_data_loader, validation_data_loader

    def _get_data(self) -> Dict[Key, np.array]:
        """
        Wrapper for collecting all data (i.e. samples, targets) used for the training
        Returns: A dictionary of all relevant tensors
        """
        learner_data = {d.SAMPLES: self._learner_observations,
                        d.TARGETS: np.zeros(self._learner_observations.shape[0]).astype(np.float32)}
        expert_data = {d.SAMPLES: self._expert_observations,
                       d.TARGETS: np.ones(self._expert_observations.shape[0]).astype(np.float32)}

        if not len(self._expert_observations) == len(self._learner_observations):  # unbalanced expert vs learner data
            learner_weight = len(self._expert_observations) / len(self._learner_observations)
            learner_data[d.WEIGHTS] = np.full(self._learner_observations.shape[0], learner_weight).astype(np.float32)
            expert_data[d.WEIGHTS] = np.ones(self._expert_observations.shape[0]).astype(np.float32)

        all_data = {k: interweave(learner_data[k], expert_data[k]) for k in learner_data.keys()}
        return all_data
