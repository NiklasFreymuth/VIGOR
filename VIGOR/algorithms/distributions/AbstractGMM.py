from algorithms.distributions.Categorical import Categorical
from algorithms.distributions.Gaussian import Gaussian
import numpy as np
import abc  # abstract base class
from util.Types import *


class AbstractGMM(abc.ABC):
    """
    Abstract GMM that only implements the mixture part and leaves the concrete implementation of the Gaussian
    open. Can not be used directly but is instead the father of the other GMM classes.
    """

    def __init__(self, weights):
        """
        Initialize the GMM with given weights. These can either be log-weights or the actual weights, but must
        in any case resemble a categorical distribution
        :param weights: Weights as a list [w1, w2, w3...]
        """
        if np.isclose(np.sum(weights), 1):
            # model weights as a categorical distribution
            assert not any([x == 0 for x in weights]), "Can not have component with weight zero"
            self._weight_distribution = Categorical(log_probabilities=np.log(weights))
        elif np.isclose(np.sum(np.exp(weights)), 1):
            self._weight_distribution = Categorical(log_probabilities=weights)
        else:
            raise AssertionError(
                "Weights are not a valid (log-)distribution: {}, sum={}".format(weights, np.sum(weights)))
        self.components = None

        ### recording utility ###
        self.previous_component_ids = set()

    @abc.abstractmethod
    def log_density(self, samples: np.ndarray):
        raise NotImplementedError("Use instance of GMM instead of the abstract one")

    @abc.abstractmethod
    def sample(self, num_samples: int):
        raise NotImplementedError("Use instance of GMM instead of the abstract one")

    def sample_per_component(self, num_samples_per_component: int) -> np.array:
        return np.array([component.sample(num_samples=num_samples_per_component) for component in self.components])

    def log_density_per_component(self, samples: np.array) -> np.array:
        return np.array([component.log_density(samples) for component in self.components])

    @property
    def weights(self) -> np.ndarray:
        return self.weight_distribution.probabilities

    @property
    def log_weights(self) -> np.array:
        return self._weight_distribution.log_probabilities

    @property
    def means(self) -> np.array:
        return np.array([c.mean for c in self.components])

    @property
    def covars(self) -> np.array:
        return np.array([c.covar for c in self.components])

    @property
    def component_entropy(self) -> float:
        return self._weight_distribution.entropy

    @property
    def num_components(self) -> int:
        return len(self.components)

    @property
    def weight_distribution(self) -> Categorical:
        return self._weight_distribution

    @property
    def dimension(self) -> int:
        return self.components[0].mean.shape[0]

    @property
    def components_as_dict(self) -> Dict[int, float]:
        """
        Wrapper for the probabilities_as_dict of the categorical distribution to keep the law of demeter
        :return: The weight distribution of the categorical distribution as a dictionary over ids
        """
        return self._weight_distribution.probabilities_as_dict()

    @property
    def component_ids(self) -> List[int]:
        return self._weight_distribution.ids

    def add_component(self, initial_weight, initial_mean: np.ndarray, initial_covar: np.ndarray) -> Optional[int]:
        """
       Adds a component specified by the given weight, mean and covariance
       :param initial_weight: The initial weight of the newly added component
       :param initial_mean: The initial mean of the added component
       :param initial_covar:
       :return: The id of the created component if it is successfully added, None otherwise
       """
        new_component = Gaussian(initial_mean, initial_covar)
        if new_component.mean is None:
            # Component initialization failed
            return None
        component_id = self._weight_distribution.add_entry(initial_weight)
        self.components.append(Gaussian(initial_mean, initial_covar))
        return component_id

    def merge(self, other_gmm, new_weight: float = 0.5, deletion_threshold: float = 1e-6) \
            -> Tuple[List[int], List[int]]:
        """
        Merges this GMM with another GMM by adding all components of the other GMM to this one and
        creating a new weight distribution that is a normalized concatenation of the old ones
        :param other_gmm: A GMM to merge into this one
        :param new_weight: Total component weight of the new gmm (other_gmm) after the merge. 0.5 is an even merge
        :param deletion_threshold: Weight threshold for now obsolete components. Components below this
         weight are deleted.
        :return: The component_ids for the new components added and the component_ids for the deleted components
        """
        # add components
        added_ids = self._weight_distribution.merge(other_categorical=other_gmm.weight_distribution,
                                                    new_weight=new_weight)
        self.components.extend(other_gmm.components)

        # remove now obsolete components
        weights = self._weight_distribution.probabilities
        deletable_component_positions = [idx for idx, component_weight in enumerate(weights)
                                         if component_weight < deletion_threshold]
        deleted_ids = []
        for deletable_component_position in reversed(deletable_component_positions):
            # delete backwards due to shifting indices
            deleted_ids.append(self.remove_component(position=deletable_component_position))

        if len(self.components) == 0:
            # in very rare cases it may happen that all components have a very low weight and that
            # we delete all of them in this step.
            #  Note that this requires more than 1/deletion_threshold total components, with all of them having
            # roughly the same weight
            raise ValueError("Deleted all components during merge")
        return added_ids, deleted_ids

    def remove_component(self, position):
        """
        Removes the component at the given position in the list of existing components
        :param position: The position of the component in the (ordered) list of currently existing components
        :return: The unique id of the deleted component
        """
        component_id = self._weight_distribution.remove_entry(position=position)
        del self.components[position]
        return component_id

    def get_component_id(self, position):
        return self.component_ids[position]

    def get_equiweighted_gmm(self):
        raise NotImplementedError("Use instance of GMM instead of AbstractGMM")
