import numpy as np


def calculate_a_b(ex_s, var_s):
    alphas = np.zeros_like(ex_s)
    betas = np.zeros_like(var_s)
    for i in range(3):
        for j in range(3):
            x = ex_s[i][j]
            y = var_s[i][j]
            alpha = -(x*y + x**3 - x**2) / y
            beta = (x * (y + 1) - y + x**3 - 2 * x**2 ) / y
            print(alpha, beta)
            alphas[i][j] = alpha
            betas[i][j] = beta
    return alphas, betas



# print(type(alpha_s))

if __name__  == "__main__":
    ex_s  = np.array([[0.6, 0.25, 0.15], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    var = np.array([[0.14, 0.15, 0.1], [0.15, 0.15, 0.15], [0.15, 0.15, 0.15]])
    var_s = var * var

    alphas, betas  = calculate_a_b(ex_s, var_s):
    print("\nalpha_s is:")
    print(alphas)

    print("\nbeta_s is:")
    print(betas)
