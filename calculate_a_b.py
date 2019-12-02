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
    #  (topography, vegetation, and elevation) as priori 
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

    # a 3 transition matrix of elevation
    elevation_exs = np.array([0.6, 0.25, 0.15])
    elevation_stds  = np.array([0.14, 0.15, 0.1])
    elevation_vars = elevation_stds * elevation_stds


    # calculate alpha and beta from expectation and variance
    # topography, vegetation, and elevation
    topography_alphas, topography_betas  = calculate_alphas_betas(topography_exs, topography_vars)
    vegetation_alphas, vegetation_betas  = calculate_alphas_betas(vegetation_exs, vegetation_vars)
    elevation_alphas, elevation_betas  = calculate_alphas_betas(elevation_exs, elevation_vars)
    print("\n\n###     Alpha_s list below :  ###")
    print("\ntopography_alphas = \n", topography_alphas)
    print("\nvegetation_alphas = \n", vegetation_alphas)
    print("\nelevation_alphas = \n", elevation_alphas)

    print("\n\n###     Beta_s list below :  ###")
    print("\ntopography_betas = \n", topography_betas)
    print("\nvegetation_betas = \n", vegetation_betas)
    print("\nelevation_betas = \n", elevation_betas)
    print("\n")



