import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

from get_direction_prob import theta_21to3, cal_7directions_probability, cal_7directions_probability_poster
from gibbs import Gibbs_M_H_sampler
 
from tqdm import tqdm 
import random
try:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
	pass
import cv2

from calculate_a_b import get_states_index_random
from utils import show_probility_img3D

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

flag = 0
if flag:
    samples_matrix = Gibbs_M_H_sampler(thetas)
    np.save("samples_matrix.npy", samples_matrix)



   


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



samples_matrix = np.load("samples_matrix.npy")



# posterior predictive distribution
state_transition_matrix = np.zeros((608, 608))
print("\nBuilding a 608 * 608 state transition matrix ......")
for i in tqdm(range(608)):
    thetas_random = [0 for i in range(21)]
    for k in range(21):
        thetas_random[k] = samples_matrix[random.randint(0, len(samples_matrix))-1][k]

    
    states_index_random = get_states_index_random()
    T, V, S = theta_21to3(thetas_random)
    index_7s, prob_normal_7s  = cal_7directions_probability_poster(states_index_random, i, T, V, S)
    # = prob_normal_7s = cal_7directions_probability(states_index, index_current, T, V, S)
    # print("index_7s = ", index_7s)
    # print("prob_normal_7s =", prob_normal_7s)
    for j, index in enumerate(index_7s):
        if index >=0 and index <= 607:
            state_transition_matrix[i, index] = np.array(prob_normal_7s[j])

input_vector = np.zeros((1, 608))
start_index = 353
input_vector[0, start_index] = 1


B = state_transition_matrix
np.save("transition_matrix.npy", B)
step = 200
A = input_vector
print("\nAfter 200 iterations ......")
for i in tqdm(range(step)):
    B = np.dot(np.mat(B), np.mat(B)) 
    B /= np.max(B)

probility_distribution_608s = np.dot(np.mat(A), np.mat(B))

p_data = np.array(probility_distribution_608s)
p_data = np.reshape(p_data,(32, 19))
b = p_data * 255
img = cv2.merge([b, b, b])
print("\nPosterior predictive distribution")
show_probility_img3D(img , title_name="Posterior predictive distribution")    


# ##############



