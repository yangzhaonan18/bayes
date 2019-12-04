import numpy as np
import math


from utils import Transfer_matrix, Build_State

def combine_probilities(cell_state_current, cell_state_neighbor):
    Vi = 0
    Vj = 0


def find_state_7neighbors(states, index):
    # 7neighbors_states = 0
    pass  


def judge_slop_type(cell_i, cell_j, cell_side_length=24, degree=20):
    # cell_side_length is the distance between cells
    # degree is between 0 and 360 
    # use 0 1 2 represents: uphill, no slope, and downhill
    # cell_j[3] is the elevation of the cell
    elevation_diff = cell_j[3] - cell_i[3]
    pi = 3.1416
    threshold  = cell_side_length * math.sin(degree * (pi / 180))
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



if __name__ ==  "__main__":

    T_path = "./data_TVE/T.png"
    V_path = "./data_TVE/V.png"
    E_path = "./data_TVE/E.img"
    
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

    # ###########
    # states contains all cells states, each cell's state is like [0/1/2, 0/1/2, 0/1/2, 8848.0, index]
    # ##############

    # cell_state_current = states[index]
    # cell_state_7neighbors = find_state_7neighbors(states, index)

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


    state = Build_State(T_path, V_path, E_path)
    state.show_TV_img2D()
    state.show_E_img3D()
    states = state.TVE_states

    print(states)







