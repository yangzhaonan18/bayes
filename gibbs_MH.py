
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm
import random

# 1. collect data from the photo and save as [[index, direction_index], ..., [index, direction_index]]

# 2. process function


# observations (np.ndarray, 2Xn): observations.
# alphas (np.ndarray, 21X1): alphas of thetas distribution.
# betas (np.ndarray, 21X1): betas of thetas distribution.

def process(thetas):
        """Given the 21 thetas , generate state transitions matrix (7Xn) and observations matrix (7Xn).
        Notice: The state transitions matrix should be corresponding to the observations matrix in the direction.

        Args:
                thetas (np.ndarray, 21X1): the priors.

        Returns:
                state_transitions_matrix (np.ndarray, 7Xn): the state transitions matrix after normalization.
                observations_matrix (np.ndarray, 7Xn): the observations matrix.

        """
        pass


def P(i, thetas):
        """PDF used in the M-H algorithm.
        P(theta_i) = f(z|theta)*P(theta_i), where P(theta_i) obbeys Beta distribution.

        Args:
                i (int): the index of the theta.
                thetas (np.ndarray, 21X1): the priors.

        Returns:
                probability (float): the probability.
        """
        theta_i = thetas[i]
        P_theta_i = beta(alphas[i], betas[i]).pdf(theta_i)

        state_transitions_matrix, observations_matrix = process(thetas, observations)
        f_z_given_theta = np.mean(np.sum(state_transitions_matrix*observations_matrix, axis=0))

        probability = P_theta_i * f_z_given_theta
        return probability

def Gibbs_M_H_sampler(thetas):
        """Initialize the thetas, samples with outside loop of Gibbs algorithm and inside loop of M-H algorithm.

        Args:
                thetas (np.ndarray, 21X1): the priors.
        """
        burn_in_n = 500
        samples_n = 500
        samples_matrix = np.zeros((samples_n, 21)) # save the samples_n samples, containing 21 theta values.

        for i in range((burn_in_n + samples_n)):
                # outside loop: Gibbs sampler.
                for j in range(21):
                        # inside loop: M-H sampler.
                        proposal_theta = norm.rvs(loc=thetas[j], scale=1, size=1, random_state=None)
                        import copy
                        proposal_thetas = copy.deepcopy(thetas)
                        proposal_thetas[j] = proposal_theta[0]
                        alpha = min(1, P(j, proposal_thetas) / P(j, thetas))

                        u = random.uniform(0, 1)
                        if u < alpha:
                                thetas[j] = proposal_theta[0]
                # burn in the burn_in_n samples, and save the rest samples to the samples_matrix.
                if i >= burn_in_n:
                        samples_matrix[i-burn_in_n, :] = thetas

        return samples_matrix
