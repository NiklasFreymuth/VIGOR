import numpy as np
from algorithms.auxiliary.ITPS import ITPS


class MoreGaussian(ITPS):
    """
    Updates a multinomial Gaussian with full covariance matrix in closed form via the
    MORE updates
    """

    def __init__(self, dim, eta_offset, omega_offset, constrain_entropy):
        super().__init__(eta_offset, omega_offset, constrain_entropy)

        self._dim = dim
        self._dual_const_part = dim * np.log(2 * np.pi)
        self._entropy_const_part = 0.5 * (self._dual_const_part + dim)

    def more_step(self, eps: float, beta, old_dist, reward_surrogate):
        """
        Performs a single step of MORE for the given distribution using a quadratic reward surrogate
        Args:
            eps: The kl bound used to constraint the update of the distribution
            beta:
            old_dist: Distribution to perform an update step on
            reward_surrogate: Quadratic surrogate of the reward to update the distribution on

        Returns:

        """
        self._kl_bound = eps  # the kl bound
        self._beta = beta
        self._succ = False

        self._old_lin_term = old_dist.lin_term
        self._old_precision = old_dist.precision
        self._old_mean = old_dist.mean
        self._old_chol_precision_t = old_dist.chol_precision.T

        self._reward_lin_term = reward_surrogate.lin_term
        self._reward_quad_term = reward_surrogate.quad_term

        old_logdet = old_dist.covar_logdet
        self._old_term = -0.5 * (np.dot(self._old_lin_term, self._old_mean) + self._dual_const_part + old_logdet)
        self._kl_const_part = old_logdet - self._dim

        try:
            opt_eta, opt_omega = self.opt_dual()  # optimizing eq. 4 of the more equations
            new_lin, new_precision = self._get_new_natural_params(opt_eta + self._eta_offset,
                                                                  opt_omega + self._omega_offset)
            new_covar = np.linalg.inv(new_precision)
            new_mean = new_covar @ new_lin
            self._succ = True
            return new_mean, new_covar
        except Exception as e:
            # optimization may fail for some components
            print(e)
            self._succ = False
            return None, None

    def _get_new_natural_params(self, eta: float, omega: float):
        new_lin = (eta * self._old_lin_term + self._reward_lin_term) / (eta + omega)
        new_precision = (eta * self._old_precision + self._reward_quad_term) / (eta + omega)
        return new_lin, new_precision

    def _dual(self, eta_omega, grad):
        eta, eta_plus_offset, omega, omega_plus_offset = self._update_eta_omega(eta_omega)

        new_lin, new_precision = self._get_new_natural_params(eta=eta_plus_offset, omega=omega_plus_offset)
        try:
            new_covar = np.linalg.inv(new_precision)
            new_chol_covar = np.linalg.cholesky(new_covar)

            new_mean = new_covar @ new_lin
            new_logdet = 2 * np.sum(np.log(np.diagonal(new_chol_covar) + 1e-25))

            dual = eta * self._kl_bound - omega * self._beta + eta_plus_offset * self._old_term
            dual += 0.5 * (eta_plus_offset + omega_plus_offset) * (
                        self._dual_const_part + new_logdet + np.dot(new_lin, new_mean))

            trace_term = np.sum(np.square(self._old_chol_precision_t @ new_chol_covar))
            kl = self._kl_const_part - new_logdet + trace_term
            diff = self._old_mean - new_mean
            kl = 0.5 * (kl + np.sum(np.square(self._old_chol_precision_t @ diff)))

            grad[0] = self._kl_bound - kl
            grad[1] = (self._entropy_const_part + 0.5 * new_logdet - self._beta) if self._constrain_entropy else 0.0
            self._grad[:] = grad
            return dual

        except np.linalg.LinAlgError as e:
            grad[0] = -1.0
            grad[1] = 0.0
            self._grad[:] = grad
            return 1e12
