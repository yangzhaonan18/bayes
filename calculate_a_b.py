import numpy as np
import math


from utils import Transfer_matrix, Build_State, find_7index

def combine_probilities(cell_state_current, cell_state_neighbor):
    Vi = 0
    Vj = 0





def find_state_7neighbors(states, index):
    # 7neighbors_states = 0
    pass  

# judge_slop_type(current_E_value, next_E_value)
def judge_slop_type(current_E_value, next_E_value, between_cells_distance=20, degree=15):
    # between_cells_distance is the distance between cells
    # degree is between 0 and 360 
    # use 0 1 2 represents: uphill, no slope, and downhill
    elevation_diff = int(next_E_value) - int(current_E_value)
    # print("elevation_diff = ", elevation_diff)  # 5 m
    pi = 3.1416
    threshold  = between_cells_distance * math.sin(degree * (pi / 180))
    print("elevation_diff threshold = ", threshold)  # 8
    if elevation_diff > threshold:
        slop_type = 0  # uphill
    elif elevation_diff >= - threshold and  elevation_diff <= threshold:
        slop_type = 1  # no slope
    elif elevation_diff <  - threshold:
        slop_type = 2  # downhill
    return slop_type


def  normalize_probilities(probability_list):
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
                                [0.4, 0.4, 0.2]])
    vegetation_stds = np.array([[0.14, 0.15, 0.1], 
                               [0.15, 0.15, 0.15], 
                               [0.15, 0.15, 0.15]])
    vegetation_vars = vegetation_stds * vegetation_stds

    # a 3 transition matrix of slope
    slope_exs = np.array([0.6, 0.25, 0.15])
    slope_stds  = np.array([0.14, 0.15, 0.1])
    slope_vars = slope_stds * slope_stds

    transfer_matrix = Transfer_matrix(topography_exs, topography_vars, 
        vegetation_exs, vegetation_vars, slope_exs, slope_vars)
    sample = transfer_matrix.find_beta_sample("topography", 0, 0)
    print("sample = ", sample)

    

    Ti, Tj = 2, 2
    Vi, Vj = 2, 2
    Si = 2


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

    print(states_ij[0][0])
    print(states_ij[10][10])
    print(states_ij[31][18])
    print(states_index[0])

    index_input = 100

    ij_7s, index_7s = find_7index(index_input) # cell index is 100

    states_7s = find_7state(states_index, index_7s)

    current_state = states_7s[0]
    for i in range(1, 7, 1):
        next_state = states_7s[i]
        current_T_type, next_T_type = current_state[0], next_state[0]
        current_V_type, next_V_type = current_state[1], next_state[1]
        current_E_value, next_E_value = current_state[2], next_state[2]
        print("current_E_value, next_E_value = ", current_E_value, next_E_value)

        slop_type = judge_slop_type(current_E_value, next_E_value)

        p_T = transfer_matrix.find_beta_sample("topography", current_T_type, next_T_type)
        p_V = transfer_matrix.find_beta_sample("vegetation", current_V_type, next_V_type)
        p_S = transfer_matrix.find_beta_sample("slope", slop_type)
        print(p_T, p_V, p_S)


        








    # print("\nstates_7s = \n", states_7s)

    # print("\ninput_index = ", input_index)
    # print("output_ij = ", ij_7s)
    # print("index_7s = ", index_7s)







