import numpy as np
from scipy.spatial import distance


class Gaussian:

    def __init__(self, mean: np.array, covar: np.array):
        if isinstance(mean, list):
            mean = np.array(mean)
        if isinstance(covar, list):
            covar = np.array(covar)
        assert mean.ndim == 1, "Must have mean as a vector. Given shape '{}' instead".format(mean.shape)
        assert covar.ndim == 2, "Must have 2-dimensional covariance. Given shape '{}' instead".format(covar.shape)
        self._mean = None
        self._dimension = mean.shape[-1]
        self.update_parameters(mean=mean, covar=covar)

        self._total_drawn_samples = 0

    def density(self, samples: np.array):
        return np.exp(self.log_density(samples))

    def log_density(self, samples: np.array, *args, **kwargs) -> np.array:
        """
        Calculates the log density of a given list of samples, where each sample is represented by a list of parameters
        :param samples: A list of samples, where each sample is a list of points of the appropriate dimensionality:
        [sample1, ..., sample_n] with sample_i = [sample_i1, ..., sample_id]
        :return:  The log density for each sample, i.e. [density1, ...density_n]
        """
        assert np.ndim(samples) == 2, "Samples must be a list of samples where each sample is a lists of parameters"
        assert samples.shape[-1] == self._dimension, "Dimension of samples must match"
        norm_term = self._dimension * np.log(2 * np.pi) + self.covar_logdet

        diff = samples - self._mean
        exp_term = np.sum(np.square(diff @ self._chol_precision), axis=-1)
        _log_density = -0.5 * (norm_term + exp_term)
        return _log_density

    def log_likelihood(self, samples):
        return np.mean(self.log_density(samples))

    def sample(self, num_samples: int) -> np.array:
        """
        Draws samples from a uniform Gaussian and then transforms them using the mean and covariance of this Gaussian
        Args:
            num_samples: Number of samples to draw

        Returns: An array [samples, dimension-wise values]

        """
        eps = np.random.normal(size=[num_samples, self._dimension])
        self._total_drawn_samples = self._total_drawn_samples + num_samples
        return self._mean + eps @ self._chol_covar.T

    @property
    def entropy(self) -> np.array:
        if self._entropy is None:
            self._entropy = 0.5 * (self._dimension * np.log(2 * np.pi * np.e) + self.covar_logdet)
        return self._entropy

    def kl(self, other):
        trace_term = np.sum(np.square(other.chol_precision.T @ self._chol_covar))
        kl = other.covar_logdet - self.covar_logdet - self._dimension + trace_term
        diff = other.mean - self._mean
        kl = kl + np.sum(np.square(other.chol_precision.T @ diff))
        return 0.5 * kl

    @property
    def covar_logdet(self) -> np.array:
        if self._covar_logdet is None:  # has recently been reset
            self._covar_logdet = 2 * np.sum(np.log(np.diagonal(self._chol_covar) + 1e-25))
        return self._covar_logdet

    def update_parameters(self, mean: np.array, covar: np.array):
        try:
            chol_covar = np.linalg.cholesky(covar)
            inverse_chol_covar = np.linalg.inv(chol_covar)
            precision = inverse_chol_covar.T @ inverse_chol_covar
            self._chol_covar = chol_covar
            self._precision = precision
            self._chol_precision = np.linalg.cholesky(precision)  # precision = standard deviation ^-1
            self._lin_term = precision @ mean  # also the natural mean

            self._mean = mean
            self._covar = covar

            # reset and calculate things once we actually need them
            self._inverse_covar = None
            self._covar_logdet = None
            self._entropy = None

        except Exception as e:
            print("Gaussian Paramameter update rejected:", e)
            return

    def mahalanobis_distance(self, samples: np.array) -> np.array:
        """
        Gives the mahalanobis distance for each sample
        Args:
            samples: A numpy array of samples

        Returns: The mahalnobis distance for each sample
        """
        mahalanobis_distances = np.array(
            [distance.mahalanobis(u=sample, v=self.mean, VI=self.inverse_covar) for sample in samples])
        return mahalanobis_distances

    @property
    def mean(self):
        return self._mean

    @property
    def inverse_covar(self):
        if self._inverse_covar is None:
            if self.covar.shape[-1] == 1:
                self._inverse_covar = 1 / self.covar
            else:
                self._inverse_covar = np.linalg.inv(self.covar)
        return self._inverse_covar

    @property
    def covar(self):
        return self._covar

    @property
    def lin_term(self):
        return self._lin_term

    @property
    def precision(self):
        return self._precision

    @property
    def chol_covar(self):
        return self._chol_covar

    @property
    def chol_precision(self):
        return self._chol_precision

    @property
    def total_drawn_samples(self):
        return self._total_drawn_samples
