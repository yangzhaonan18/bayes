import numpy as np
import math
import cv2

from utils import Transfer_matrix, Build_State, find_7index, show_probility_img3D
from tqdm import tqdm


# judge_slop_type(current_E_value, next_E_value)
def judge_slop_type(current_E_value, next_E_value, between_cells_distance=24, degree=20):
    # between_cells_distance is the distance between cells
    # degree is between 0 and 360 
    # use 0 1 2 represents: uphill, no slope, and downhill
    elevation_diff = int(next_E_value) - int(current_E_value)
    # print("elevation_diff = ", elevation_diff)  # 5 m
    pi = 3.1416
    threshold  = between_cells_distance * math.sin(degree * (pi / 180))
    # print("elevation_diff threshold = ", threshold)  # 8
    if elevation_diff > threshold:
        slop_type = 0  # uphill
    elif elevation_diff >= - threshold and  elevation_diff <= threshold:
        slop_type = 1  # no slope
    elif elevation_diff <  - threshold:
        slop_type = 2  # downhill
    return slop_type


def normalize_probilities(probability_list):
    # Normalize the probability values, make sure the sum of the probabilities is 1
    length = len(probability_list)
    probability_list_new = [0 for i in range(length)]
    total = 0
    for i in range(length):
        total += probability_list[i]
    for i in range(length):
        probability_list_new[i] = probability_list[i] / total 
    return probability_list_new


def find_7state(states_index, index_7s):
    states_7s = []
    for index in  index_7s:
        if index >= 0 :
            states_7s.append(states_index[index])
        else:
            # Set the region outside the map to state type 0 type 0  and 0  elevation
            states_7s.append([0, 0, 0])

    return np.array(states_7s, dtype=np.uint8)



def cal_7directions_probability(states_index, index_current):
    """
    input :
        states_index(list):   [ [0, 0, elevation], [0, 0, elevation] ....]
        index_current(int): 353

    return :
        index_7s(list):  [100, 138, 120, 82, 62, 81, 119]
        prob_normal_7s(list): [0.1 0.1. 0.2 ....]
        directions_7s_01(list): [ 0 0 1 0 0 ......]
        max_p(float): = 0.7

    """
    ij_7s, index_7s = find_7index(index_current) # cell index is 100
    states_7s = find_7state(states_index, index_7s)
    current_state = states_7s[0]
    direction_probilities = [ 0 for i in range(7)]

    for i in range(0, 7, 1):
        next_state = states_7s[i]
        current_T_type, next_T_type = current_state[0], next_state[0]
        current_V_type, next_V_type = current_state[1], next_state[1]
        current_E_value, next_E_value = current_state[2], next_state[2]
        # print("current_E_value, next_E_value = ", current_E_value, next_E_value)
        slop_type = judge_slop_type(current_E_value, next_E_value)

        p_T = transfer_matrix.find_theta_sample("topography", current_T_type, next_T_type)
        p_V = transfer_matrix.find_theta_sample("vegetation", current_V_type, next_V_type)
        p_S = transfer_matrix.find_theta_sample("slope", slop_type)

        combine_probilities = p_T * p_V * p_S
        direction_probilities[i] = combine_probilities 
    
    direction_probilities_normalized = normalize_probilities(direction_probilities)  # (7,)
    prob_normal_7s = direction_probilities_normalized
    
    max_direction = prob_normal_7s.index(max(prob_normal_7s))
    directions_7s_01 = [0 for i in range(7)]
    directions_7s_01[max_direction] = 1
    max_p = prob_normal_7s[max_direction]
    # print("\n # # # # # # # # # #  ")
    # print("\nindex_7s = \n", index_7s)
    # print("\n7 direction probilities after normalized = \n", prob_normal_7s)
    # print("\ndirections_01 = \n", directions_7s_01)
    # print("\nmax_p = \n", max_p)   
    return index_7s, prob_normal_7s, directions_7s_01, max_p


 

def prior_predictive_distribution_copy(states_index, index_current=353, loop=200):

    probility_distribution_608s = [0 for i in range(608)]
    probility_distribution_608s[index_current] = 1
    for i in range(loop):
        for index_current in range(608):
            p_i = probility_distribution_608s[index_current]
            if  p_i != 0:
                print("p_i = ", p_i)
                # input("sdf")
                index_7s, prob_normal_7s, directions_7s_01, max_p = cal_7directions_probability(states_index, index_current)
                # directions_number = directions_7s_01.index(max(directions_7s_01))
                conditions_probility_608s = [0 for i in range(608)]
                for i in range(len(index_7s)):
                    next_index = index_7s[i]
                    next_p = prob_normal_7s[i] 
                    # print("next_index = ", next_index)
                    if next_index >= 0 and next_index <= 607:
                        conditions_probility_608s[next_index] = next_p
                        # print("prob_normal_7s[i]", prob_normal_7s[i])
                    else:
                        print(" next_index out of range =", next_index)
                join_probability  =  [x * p_i for x in conditions_probility_608s] 
                probility_distribution_608s[index_current] = 0
                for i in range(len(join_probability)):
                    probility_distribution_608s[i] +=  join_probability[i]

    print("....")
    for i, item in enumerate(probility_distribution_608s):
        if item != 0:
            print(i, item)    

    print(sum(probility_distribution_608s))
    print(len(probility_distribution_608s))
    print(len(join_probability))
    p_data = np.array(probility_distribution_608s)
    p_data = np.reshape(p_data,(32, 19))
    b = p_data
    img = cv2.merge([b, b, b])
    show_probility_img3D(img)    



def get_states_index_random():
    T_path = "./data_TVE/T.png"
    V_path = "./data_TVE/V.png"
    E_path = "./data_TVE/E.png" 
    build_State = Build_State(T_path, V_path, E_path)
    states_index = build_State.TVE_states_index
    return states_index


if __name__ ==  "__main__":

    
    # ######
    # Experts give the expectation  and variance of the three domains
    # (topography, vegetation, and slope) as priori 
    # ######
    # a 3x3 transition matrix of topography
    topography_exs  = np.array([[0.6, 0.25, 0.15], 
                                [0.5, 0.3, 0.2], 
                                [0.4, 0.4, 0.2]])
    topography_stds = np.array([[0.14, 0.15, 0.1], 
                               [0.15, 0.15, 0.15], 
                               [0.15, 0.15, 0.15]])
    topography_vars = topography_stds * topography_stds

    # a 3x3 transition matrix of vegetation
    vegetation_exs  = np.array([[0.6, 0.25, 0.15], 
                                [0.5, 0.3, 0.2], 
                                [0.6, 0.4, 0.2]])
    vegetation_stds = np.array([[0.14, 0.15, 0.1], 
                               [0.15, 0.15, 0.15], 
                               [0.15, 0.15, 0.15]])
    vegetation_vars = vegetation_stds * vegetation_stds

    # a 3 transition matrix of slope
    slope_exs = np.array([0.2, 0.3, 0.5])
    slope_stds  = np.array([0.15, 0.14, 0.14])
    slope_vars = slope_stds * slope_stds

    transfer_matrix = Transfer_matrix(topography_exs, topography_vars, 
        vegetation_exs, vegetation_vars, slope_exs, slope_vars)
    
    # All the parameters
    alpha_21s =  np.array(transfer_matrix.alpha_21s)
    beta_21s =  np.array(transfer_matrix.beta_21s)

    theta_sample_21s = np.array(transfer_matrix.theta_sample_21s())
   
    print("\nalpha_21s =\n", alpha_21s.shape)
    print("\nbeta_21s =\n", beta_21s.shape)
    print("\ntheta_sample_21s = \n", theta_sample_21s.shape)
    np.save("alphas.npy", alpha_21s)
    np.save("betas.npy", beta_21s)
    np.save("thetas.npy", theta_sample_21s)
 

    # TVE
    T_path = "./data_TVE/T.png"
    V_path = "./data_TVE/V.png"
    E_path = "./data_TVE/E.png"
    # 
    # Use TVE information to build a status message containing 607 cells
    # 

    build_State = Build_State(T_path, V_path, E_path)
    # build_State.show_TV_img2D()
    build_State.show_E_img3D()
    # states_ij = build_State.TVE_states_ij
    states_index = build_State.TVE_states_index
    print("\nThe  state of each cell is as follows:")
    print(states_index)
    print("\nThe shape of state is:")
    print(states_index.shape) 
  
    np.save("states_index.npy", states_index)
    




 
 
    # 
    observe_indexs = [353, 371, 390, 370, 351, 313, 293, 312, 
                        330, 368, 406, 425, 463, 482, 464, 445, 465,
                        484, 504, 485, 467, 486, 506, 525, 507, 488, 470,
                        451, 433, 395, 357, 319, 300, 282, 244, 206, 186, 
                        167, 147, 128, 90,
                        ]

    index_direction = [ [ 0 , 0] for i in range(len(observe_indexs) - 1)]
    for i  in range(len(observe_indexs) -1):
        index_7s, _, directions_7s_01, _ = cal_7directions_probability(states_index, index_current=observe_indexs[i])
        next_direction = index_7s.index(observe_indexs[i + 1])
        index_direction[i][0] = observe_indexs[i]
        index_direction[i][1] = next_direction
    index_direction = np.array(index_direction)

    # print("index_direction = ", index_direction.shape)
    np.save("observations.npy", index_direction)  # [[353, 5]]
    # print("index_direction, = ", index_direction)
    # print(input("asdf"))
    # print("666") 






    # prior_predictive_distribution(states_index, index_current=353, loop=20)


    # resample from  each iteration
    # Prior predictive distribution
    if True:
        state_transition_matrix = np.zeros((608, 608))
        print("\nBuilding a 608 * 608 state transition matrix ......")
        for i in tqdm(range(608)):
            states_index_random = get_states_index_random()
            index_7s, prob_normal_7s, directions_7s_01, max_p  = cal_7directions_probability(states_index_random, 
                 index_current=i)
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
        print("\nPrior predictive distribution")
        show_probility_img3D(img, title_name="Prior predictive distribution")    
        


