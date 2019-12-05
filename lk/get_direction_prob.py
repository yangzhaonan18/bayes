#!/usr/bin/env python

import numpy as np

def theta_21to3(thetas):
        T, V, S = np.ones((9, 1)), np.ones((9, 1)), np.ones((3, 1))
        T = np.array(thetas[0: 8]).reshape((3, 3))
        V = np.array(thetas[9: 17]).reshape((3, 3))
        S = np.array(thetas[18: ])

        return T, V, S

def ij2index(i, j):
    W_num =19
    # Cells that are not on the map are set to a negative index
    if i < 0 or i > 31 or j < 0  or j > 18 :  
        return -abs(W_num * i + j)
    return W_num * i + j

def index2ij(index_cell):
    W_num = 19
    return  index_cell // W_num, index_cell % W_num


def find_7index(index):
    """    
    return:
        ij_7s : 7x2 like  [[i, j], [i,j] ....]
        index_7s : 7 index
    """
    H_num = 32
    W_num = 19
    max_inde = 607  # have 608 cell
    if index > 607 or index < 0:
        print("Input index error [0, 607]!!!")
    ij_7s = [[-1, -1] for i in range(7)]
    index_7s = [[-1] for i in range(7)]
    i, j = index2ij(index)

    # ##########
    # if index out of range,  set -1
    # ########
    if i % 2 == 1:
        ij_7s[0][0], ij_7s[0][1] = i, j
        ij_7s[1][0], ij_7s[1][1] =  i + 2, j  
        ij_7s[2][0], ij_7s[2][1] =  i + 1, j + 1 
        ij_7s[3][0], ij_7s[3][1] =  i - 1, j + 1 
        ij_7s[4][0], ij_7s[4][1] =  i - 2, j
        ij_7s[5][0], ij_7s[5][1] =  i - 1, j
        ij_7s[6][0], ij_7s[6][1] =  i + 1, j
    elif i % 2 == 0:
        ij_7s[0][0], ij_7s[0][1] = i, j
        ij_7s[1][0], ij_7s[1][1] =  i + 2, j  
        ij_7s[2][0], ij_7s[2][1] =  i + 1, j 
        ij_7s[3][0], ij_7s[3][1] =  i - 1, j
        ij_7s[4][0], ij_7s[4][1] =  i - 2, j
        ij_7s[5][0], ij_7s[5][1] =  i - 1, j - 1
        ij_7s[6][0], ij_7s[6][1] =  i + 1, j - 1
    # print("input index = ", index)
    # print("current i j index is = ", i, j)
    # print("output 7 index = ", ij_7s)
    for i in range(len(ij_7s)):
        index_7s[i] = ij2index(ij_7s[i][0], ij_7s[i][1])
    return ij_7s, index_7s

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


def cal_7directions_probability(states_index, index_current, T, V, S):
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

        p_T = T[current_T_type, next_T_type]
        p_V = V[current_V_type, next_V_type]
        p_S = S[slop_type]

        combine_probilities = p_T * p_V * p_S
        direction_probilities[i] = combine_probilities 
    
    direction_probilities_normalized = normalize_probilities(direction_probilities)  # (7,)
    prob_normal_7s = direction_probilities_normalized

    return prob_normal_7s
