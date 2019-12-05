import numpy as np
import math
import cv2



from utils import Transfer_matrix, Build_State, find_7index, show_probility_img3D

 

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

    return :
        index_7s: [100, 138, 120, 82, 62, 81, 119]
        directions:  [0, 0, 0, 0, 0, 1, 0]
        likelihood: 0.2006794611459864

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

        p_T = transfer_matrix.find_beta_sample("topography", current_T_type, next_T_type)
        p_V = transfer_matrix.find_beta_sample("vegetation", current_V_type, next_V_type)
        p_S = transfer_matrix.find_beta_sample("slope", slop_type)
        
        combine_probilities = p_T * p_V * p_S
        direction_probilities[i] = combine_probilities 
    
    direction_probilities_normalized = normalize_probilities(direction_probilities)  # (7,)
    prob_normal_7s = direction_probilities_normalized
    
    max_direction = prob_normal_7s.index(max(prob_normal_7s))
    directions_7s_01 = [0 for i in range(7)]
    directions_7s_01[max_direction] = 1
    max_p = prob_normal_7s[max_direction]
    print("\n # # # # # # # # # #  ")
    print("\nindex_7s = \n", index_7s)
    print("\n7 direction probilities after normalized = \n", prob_normal_7s)
    print("\ndirections_01 = \n", directions_7s_01)
    print("\nmax_p = \n", max_p)
    

    return index_7s, prob_normal_7s, directions_7s_01, max_p


def prior_predictive_distribution(states_index, index_current=353, loop=200):

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


if __name__ ==  "__main__":

    T_path = "./data_TVE/T.png"
    V_path = "./data_TVE/V.png"
    E_path = "./data_TVE/E.png"
    
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
    slope_exs = np.array([0.2, 0.4, 0.4])
    slope_stds  = np.array([0.15, 0.14, 0.3])
    slope_vars = slope_stds * slope_stds

    transfer_matrix = Transfer_matrix(topography_exs, topography_vars, 
        vegetation_exs, vegetation_vars, slope_exs, slope_vars)
    sample = transfer_matrix.find_beta_sample("topography", 0, 0)
    print("sample = ", sample)

    

    # Ti, Tj = 2, 2
    # Vi, Vj = 2, 2
    # Si = 2


    # index = 0

  
    # #######################################
    # cell_state[0] use 0, 1, 2 represents topography:  lake, plain, hill 
    # cell_state[1] use 0, 1, 2 represents vegetation density: sparse, medium,dense
    # cell_state[2] use float value represents elevation
    # cell_state[3] use int represents the index of this cell
    # #######################################

    # fai = [0 for i in range(7)]
    # for cell_state_neighbor in cell_state_7neighbors:
    #     fai[i] = combine_probilities(cell_state_current, cell_state_neighbor)
    # fai = normalize_probilities(fai)


    build_State = Build_State(T_path, V_path, E_path)
    # build_State.show_TV_img2D()
    # build_State.show_E_img3D()
    states_ij = build_State.TVE_states_ij
    states_index = build_State.TVE_states_index


    # prior_predictive_distribution(states_index, index_current=353, loop=200)


    transfer_matrix.topography_alphas
    transfer_matrix.topography_betas
    transfer_matrix.vegetation_alphas
    transfer_matrix.vegetation_betas
    transfer_matrix.slope_alphas
    transfer_matrix.slope_betas


   
    index_7s, prob_normal_7s, directions_7s_01, max_p = cal_7directions_probability(states_index, index_current=353)

    observe_indexs = [353, 371, 390, 370, 351, 313, 293, 312, 
                        330, 368, 406, 425, 463, 482, 464, 445, 465,
                        484, 504, 485, 467, 486, 508, 525, 507, 488, 470]
    for i  in range(len(observe_indexs) -1):
        flag = 1
        index_7s, _, directions_7s_01, _ = cal_7directions_probability(states_index, index_current=observe_indexs[i])
         











    # print("\nstates_7s = \n", states_7s)
    # print("\ninput_index = ", input_index)
    # print("output_ij = ", ij_7s)
    # print("index_7s = ", index_7s)







