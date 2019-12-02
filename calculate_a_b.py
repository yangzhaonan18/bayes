import numpy as np


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

    print(V_sample)
    print(T_sample)
    print(S_sample)


