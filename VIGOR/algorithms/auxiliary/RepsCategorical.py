import numpy as np
from algorithms.auxiliary.ITPS import ITPS
from util.Functions import logsumexp
from util.Normalization import log_normalize


class RepsCategorical(ITPS):
    """
    Reps: Relative Entropy Policy Search
    Responsible for learning the weight distribution of the GMM for the marginal EIM
    """

    def reps_step(self, eps, beta, old_dist, rewards):
        """

        Args:
            eps:
            beta:
            old_dist:
            rewards:

        Returns: The logarithm of the new params.

        """
        self._kl_bound = eps
        self._beta = beta
        self._old_log_prob = old_dist.log_probabilities
        self._rewards = rewards

        try:
            opt_eta, opt_omega = self.opt_dual()
            new_log_params = self._new_log_params(opt_eta + self._eta_offset, opt_omega + self._omega_offset)
            self._succ = True
            return new_log_params
        except Exception:
            self._succ = False
            return None

    def _new_log_params(self, eta, omega):
        omega = omega if self._constrain_entropy else self._omega_offset
        new_log_weights = (eta * self._old_log_prob + self._rewards) / (eta + omega)  # logsumexp trick
        new_log_weights = log_normalize(new_log_weights)
        new_log_weights -= logsumexp(new_log_weights)  # normalize in log-space
        return new_log_weights

    def _dual(self, eta_omega, grad):
        """
        A lot of math for the REPS step
        :param eta_omega: a tuple (eta, omega)
        :param grad:
        :return:
        """
        eta, eta_plus_offset, omega, omega_plus_offset = self._update_eta_omega(eta_omega)

        t1 = (eta_plus_offset * self._old_log_prob + self._rewards) / (eta_plus_offset + omega_plus_offset)
        #  one times(eta + omega) in denominator  missing
        t1_de = (omega_plus_offset * self._old_log_prob - self._rewards) / (eta_plus_offset + omega_plus_offset)
        #  t1_do = -t1 with one times (eta+omega) in denominator missing

        t1_max = np.max(t1)
        exp_t1 = np.exp(t1 - t1_max)
        sum_exp_t1 = np.sum(exp_t1) + 1e-25
        t2 = t1_max + np.log(sum_exp_t1)

        #  factor of exp(t1_max) is still missing in sum_exp_t1
        inv_sum = (1 / sum_exp_t1)
        #  missing factors of exp(t1_max) in both inv_sum and exp_t1, cancel out here.
        t2_de = inv_sum * np.sum(t1_de * exp_t1)
        t2_do = - inv_sum * np.sum(t1 * exp_t1)  # -t2 =  t2_do

        grad[0] = self._kl_bound + t2 + t2_de  # missing factor in t2_de cancels out with missing factor here
        #  missing factor in t2_do cancels out with missing factor here
        grad[1] = - self._beta + t2 + t2_do if self._constrain_entropy else 0.0

        self._grad[:] = grad

        dual = eta * self._kl_bound - omega * self._beta + (eta_plus_offset + omega_plus_offset) * t2
        return dual
