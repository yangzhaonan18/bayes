import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

from get_direction_prob import theta_21to3, cal_7directions_probability
from gibbs import Gibbs_M_H_sampler


alphas = np.load("./data_npy/alphas.npy")
betas = np.load("./data_npy/betas.npy")
thetas = np.load("./data_npy/thetas.npy")
# observations = np.load("./data_npy/observations.npy")
# states_index = np.load("./data_npy/states_index.npy")

# T, V, S = theta_21to3(thetas)
# n = observations.shape[0]
# state_transitions_matrix, observations_matrix = np.ones((n, 7)), np.ones((n, 7))

# for i in range(n):
#     index_current = observations[i, 0]
#     prob_normal_7s = cal_7directions_probability(states_index, index_current, T, V, S)
#     state_transitions_matrix[i, :] = prob_normal_7s

#     index_next = observations[i, 1]
#     tmp_observation = np.zeros((7))
#     tmp_observation[index_next] = 1
#     observations_matrix[i, :] = tmp_observation

samples_matrix = Gibbs_M_H_sampler(thetas)
np.save("samples_matrix.npy",samples_matrix)


plt.figure(figsize=(15, 7))
for i in range(21):
    plt.subplot(7, 3, i+1)
    plt.hist(samples_matrix[:, i], bins=100, color='red',histtype='stepfilled',alpha=0.75)
    a, b = float(alphas[i]), float(betas[i])
    x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
    plt.plot(x, beta.pdf(x, a, b)*100, 'b-', lw=5, alpha=0.6, label='beta pdf')
    plt.title('%s' % i)

plt.savefig('out.png')
plt.show()