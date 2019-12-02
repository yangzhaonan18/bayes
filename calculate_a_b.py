import numpy as np
import math



def judge_slop_type(cell_i, cell_j, cell_side_length=24, degree=20):
    # use 0 1 2 represents: uphill, no slope, and downhill
    # cell_j[3] is the elevation of the cell
    elevation_diff = cell_j[3] - cell_i[3]
    pi = 3.1416
    threshold  = cell_side_length * math.sin(20 * (pi / 180))
    if elevation_diff > threshold:
        slop_type = 0  # uphill
    elif elevation_diff >= - threshold and  elevation_diff <= threshold:
        slop_type = 1  # no slope
    elif elevation_diff <  - threshold; 
        slop_type = 2  # downhill
    return slop_type


def  normalize_probability(p0, p1, p2):
    total = p0 + p1 + p2
    return p0 / total, p1 / total, p2 / total


def calculate_alphas_betas(ex_s, var_s):
    alphas = np.zeros_like(ex_s)
    betas = np.zeros_like(var_s)
    dimension = len(list(alphas.shape))
    print("########")
    print("The shape of the transition matrix is", alphas.shape)
    print("dimension is ", dimension)
    

    if dimension  == 2:
        for i in range(3):
            for j in range(3):
                x = ex_s[i][j]
                y = var_s[i][j]
                alpha = -(x*y + x**3 - x**2) / y
                beta = (x * (y + 1) - y + x**3 - 2 * x**2 ) / y
                alphas[i][j] = alpha
                betas[i][j] = beta

    elif dimension  == 1:
        for i in range(3):
            x = ex_s[i]
            y = var_s[i]
            alpha = -(x*y + x**3 - x**2) / y
            beta = (x * (y + 1) - y + x**3 - 2 * x**2 ) / y
            alphas[i] = alpha
            betas[i] = beta

    return alphas, betas



# print(type(alpha_s))

if __name__  == "__main__":
    # ######
    # 
    # Experts give the expectation  and variance of the three domains
    #  (topography, vegetation, and slope) as priori 
    # 
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


    # calculate alpha and beta from expectation and variance
    # topography, vegetation, and slope
    topography_alphas, topography_betas  = calculate_alphas_betas(topography_exs, topography_vars)
    vegetation_alphas, vegetation_betas  = calculate_alphas_betas(vegetation_exs, vegetation_vars)
    slope_alphas, slope_betas  = calculate_alphas_betas(slope_exs, slope_vars)
    print("\n\n###     Alpha_s list below :  ###")
    print("\ntopography_alphas = \n", topography_alphas)
    print("\nvegetation_alphas = \n", vegetation_alphas)
    print("\nslope_alphas = \n", slope_alphas)

    print("\n\n###     Beta_s list below :  ###")
    print("\ntopography_betas = \n", topography_betas)
    print("\nvegetation_betas = \n", vegetation_betas)
    print("\nslope_betas = \n", slope_betas)
    print("\n")


    Ti, Tj = 2, 2
    Vi, Vj = 2, 2
    Si = 2

    # T_alpha and T_beta is a Beta distribution's alpha and beta of topography
    T_alpha = topography_alphas[Ti][Tj]
    T_beta = topography_betas[Ti][Tj]
    # V_alpha and V_beta is a Beta distribution's alpha and beta of elevation
    V_alpha = vegetation_alphas[Vi][Vj]
    V_beta = vegetation_betas[Vi][Vj]
    # S_alpha and S_beta is a Beta distribution's alpha and beta of slope
    S_alpha =  slope_alphas[Si]
    S_beta = slope_betas[Si]

    # T_sample represents the probability of transitioning from vegetation type Ti to Tj
    T_sample = np.random.beta(T_alpha, T_beta)
    # V_sample represents the probability of transitioning from vegetation type Vi to Vj
    V_sample = np.random.beta(V_alpha, V_beta)
    # S_sample  represents the probability of following a certain local slope type Si
    S_sample = np.random.beta(S_alpha, S_beta)

    print(T_sample)
    print(V_sample)
    print(S_sample)
    cell_index = 0
    cell_state = [0, 0, 120, index]
    # cell_state[0] use 0, 1, 2 represents topography:  lake, plain, hill 
    # cell_state[1] use 0, 1, 2 represents vegetation density: sparse, medium,dense
    # cell_state[2] use float value represents elevation
    # cell_state[3] use int represents the index of this cell








    X = state




