import numpy as np
from util.Normalization import log_normalize


class Categorical:
    """
    Class for a categorical probability distribution (i.e. dice rolls)
    Used in EIM Context for the weights of the individual components.
    """

    def __init__(self, log_probabilities):
        self.log_probabilities = log_probabilities
        self._ids = list(range(len(log_probabilities)))
        self._id_count = len(log_probabilities)

    def sample(self, num_samples):
        """
        Sample num_samples entries from the distribution
        :param num_samples: number of samples
        :return: A list of samples of size num_samples
        """
        thresholds = np.expand_dims(np.cumsum(self.probabilities), 0)
        thresholds[0, -1] = 1.0
        eps = np.random.uniform(size=[num_samples, 1])
        samples = np.argmax(eps < thresholds, axis=-1)
        return samples

    def probabilities_as_dict(self) -> dict:
        """
        Convenience wrapper to return the probabilities as a dictionary of their ids
        :return:
        """
        return {self._ids[i]: self.probabilities[i] for i in range(len(self.probabilities))}

    @property
    def ids(self) -> list:
        """
        self._ids keeps track of the ids of all elements
        :return:
        """
        return self._ids

    @property
    def probabilities(self) -> np.ndarray:
        """
        Returns the probabilities of this categorical distribution.
        Note that there is no setter here since we only want to modify the probabilities in logspace
        Returns:

        """
        return self._probabilities

    @property
    def log_probabilities(self) -> np.ndarray:
        return self._log_probabilities

    @log_probabilities.setter
    def log_probabilities(self, new_log_probabilities: np.ndarray):
        """
        Sets the new log_probabilities and also updates the probabilities accordingly. Assumes that
        np.sum(np.exp(new_log_probabilities))==1 since we need a distribution at the end.
        Args:
            new_log_probabilities: The new/updated log probabilities

        Returns:

        """
        self._log_probabilities = new_log_probabilities
        probabilities = np.exp(new_log_probabilities)
        # normalize the already normalized values again to prevent precision errors.
        # because apparently that's what happens if you force rocks to think
        self._probabilities = probabilities/np.sum(probabilities)

    def add_entry(self, new_probability: float) -> int:
        """
        Add a new entry into the distribution and ensure that the sum of all probabilities remains 1.
        All other entries are normalized around the new entries.
         If the distribution was (0.2, 0.8) and a new entry with weight 0.5 is added, it becomes (0.1, 0.4, 0.5).
        :param new_probability: The new probability to be added as a component to the categorical distribution.
        This probability can either be in log-space (in which case it must be <=0), or a normal probability
        :return: The id of the added component
        """
        assert new_probability < 1 and not new_probability == 0, \
            "New probability must either be a log-probability or a probability, not '{}'".format(new_probability)
        if new_probability < 0:
            new_log_probability = new_probability
        else:
            new_log_probability = np.log(new_probability)

        new_log_probabilities = np.concatenate(
            [self.log_probabilities, new_log_probability + np.log(1 + np.exp(new_log_probability)) * np.ones(1, )],
            axis=0)
        self.log_probabilities = log_normalize(new_log_probabilities)
        self._ids.append(self._id_count)
        _id = self._id_count
        self._id_count += 1
        return _id

    def merge(self, other_categorical, new_weight=0.5):
        """
        Merge another categorical distribution into this one. The current probabilities get the specified total weight,
        the other 1-weight.
        If weight=0.5 and the current distribution is (0.2, 0.8) and a distribution (0.4, 0.6) is added, the result
        is a distribution (0.1, 0.4, 0.2, 0.3)
        :param other_categorical: The categorical distribution to merge into this one.
        :param new_weight: Weight of the new (other) distribution when being merged with the new one
        :return: A list of added ids
        """
        assert 0 < new_weight < 1, "New_weight must be in (0,1)"
        new_log_probabilities = np.concatenate(
            [self.log_probabilities + np.log(1 - new_weight), other_categorical.log_probabilities + np.log(new_weight)],
            axis=0)
        self.log_probabilities = log_normalize(new_log_probabilities)  # to prevent numerical instabilities

        added_ids = []
        for _ in other_categorical.probabilities:  # assign/create new ids
            self._ids.append(self._id_count)
            added_ids.append(self._id_count)
            self._id_count += 1
        return added_ids

    def remove_entry(self, position: int):
        """
        Remove an entry from the probability distribution. The others will be normalized accordingly to ensure
        that the distribution remains valid
        :param position: Position of the entry to be remvoed.
        :return: The id of the deleted element
        """
        if position > self._log_probabilities.size:
            raise AssertionError("Invalid Index for Categorical")
        new_log_probabilities = np.concatenate(
            [self.log_probabilities[:position], self.log_probabilities[position + 1:]], axis=0)
        self.log_probabilities = log_normalize(new_log_probabilities)
        return self._ids.pop(position)

    @property
    def entropy(self):
        """
        Return the entropy
        :return: H(self)
        """
        return - np.sum(self.probabilities * self.log_probabilities)

    def kl(self, other_categorical) -> np.ndarray:
        """
        Return the forward KL-Div. for this distribution when compared to another
        :param other_categorical: The other distribution. Must have the same dimensionality as this one.
        :return: KL(self, other)
        """
        return np.sum(self.probabilities * (self.log_probabilities - other_categorical.log_probabilities))
