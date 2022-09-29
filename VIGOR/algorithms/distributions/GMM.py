import numpy as np
from algorithms.distributions.Gaussian import Gaussian
from algorithms.distributions.AbstractGMM import AbstractGMM
from util.Functions import logsumexp
from typing import Dict


class GMM(AbstractGMM):

    def __init__(self, weights: np.array, means: np.array, covars: np.array):
        """
        Initialize the GMM with given weights, means and covariances.
        :param weights: Weights as a list [w1, w2, w3...]. Can either be the weights itself or the weights in log-space.
        In the first case, the weights must sum to 1, otherwise, their exp must sum to 1
        :param means: Means as a list of lists [[m11, m12, m13, ...], [m21, ...]]
        :param covars: Covars as a list of lists of lists [[[c111, c112, ...], [c121, c122, ...]],[c211, ...]]]
            Where each inner list of lists is a covariance matrix
        """
        super().__init__(weights=weights)
        # model each component as a Gaussian
        self.components = [Gaussian(means[i], covars[i]) for i in range(means.shape[0])]
        
        self._total_drawn_samples = 0

    def density(self, samples: np.array):
        """
        calculates the density of the GMM of a set of samples.
        :param samples: a list of positions to return the density for
        :return: a list of densities corresponding to the samples used as an input
        """
        unweighted_densities = np.stack([self.components[i].density(samples) for i in range(self.num_components)],
                                        axis=0)
        weights = np.expand_dims(self.weight_distribution.probabilities, axis=-1)  # weight each component
        densities = np.sum(weights * unweighted_densities, axis=0)
        return densities

    def log_density(self, samples):
        """
        Return the log density of the samples
        :param samples: A list of samples, where each sample is a list of parameters of the appropriate dimensionality:
        [sample1, ..., sample_n] with sample_i = [sample_i1, ..., sample_id]
        :return: The log density of the samples
        """
        assert np.ndim(samples) == 2, "Samples must be a list of samples where each sample is a lists of parameters"
        component_log_densities = np.array(
            [self.components[i].log_density(samples) for i in range(self.num_components)])
        # components without weight can be ignored

        if component_log_densities.ndim == 1:
            component_log_densities = component_log_densities[:, None]

        log_weights = np.expand_dims(self.log_weights, axis=1)
        weighted_log_densities = component_log_densities + log_weights

        log_densities = logsumexp(samples=weighted_log_densities)
        return log_densities

    def sample(self, num_samples):
        """
        draw samples from the GMM by first determining the component to draw from and then drawing from it.
        Samples are permuted after the drawing to decorrelate the samples and their origin gaussian
        :param num_samples: Number of samples to draw
        :return:
        """
        assert num_samples > 0, "Need to draw at least 1 sample"
        w_samples = self._weight_distribution.sample(num_samples)
        samples = []
        for i in range(self.num_components):
            counts = np.count_nonzero(w_samples == i)
            if counts > 0:
                samples.append(self.components[i].sample(counts))
        self._total_drawn_samples = self._total_drawn_samples + num_samples
        return np.random.permutation(np.concatenate(samples, axis=0))

    def __eq__(self, other) -> bool:
        """
        Compare if this GMM is equal to another
        Args:
            other: The other GMM

        Returns:

        """
        return np.array_equal(self.log_weights, other.log_weights) \
               and np.array_equal(self.means, other.means) \
               and np.array_equal(self.covars, other.covars)

    @property
    def parameter_dict(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary with all the parameters of the GMM. Note that the weights are returned as their logarithm,
        which is compatible with the way the GMM may be initialized.
        Returns:

        """
        return {"weights": self.log_weights, "means": self.means, "covars": self.covars}

    def get_equiweighted_gmm(self):
        """
        Returns a GMM that has the same components as this one, but a uniform weighting distribution between them.
        The new GMM is created as a value, not a reference, so the new GMM will not update if the old one changes
        Returns: A new GMM that has the same components as this one, but uniform weighting

        """
        return GMM(weights=np.repeat(1 / self.num_components,
                                     self.num_components),
                   means=self.means,
                   covars=self.covars)
    
    @property
    def total_drawn_samples(self):
        return self._total_drawn_samples + np.sum([gaussian.total_drawn_samples for gaussian in self.components])

    @property
    def is_equiweighted(self):
        return np.all(self.weights[0] == self.weights)
