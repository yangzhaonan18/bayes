import numpy as np
import math


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


# print(type(alpha_s))
class Transfer_matrix():
    def __init__(self,topography_exs, topography_vars, vegetation_exs, vegetation_vars,
        slope_exs, slope_vars):
        # calculate alpha and beta from expectation and variance
        # topography, vegetation, and slope
        self.topography_alphas, self.topography_betas  = self.__calculate_alphas_betas(topography_exs, topography_vars)
        self.vegetation_alphas, self.vegetation_betas  = self.__calculate_alphas_betas(vegetation_exs, vegetation_vars)
        self.slope_alphas, self.slope_betas  = self.__calculate_alphas_betas(slope_exs, slope_vars)
        print("\n\n###     Alpha_s list below :  ###")
        print("\ntopography_alphas = \n", self.topography_alphas)
        print("\nvegetation_alphas = \n", self.vegetation_alphas)
        print("\nslope_alphas = \n", self.slope_alphas)

        print("\n\n###     Beta_s list below :  ###")
        print("\ntopography_betas = \n", self.topography_betas)
        print("\nvegetation_betas = \n", self.vegetation_betas)
        print("\nslope_betas = \n", self.slope_betas)
        print("\n")

    def __calculate_alphas_betas(self, ex_s, var_s):
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
                    alphas[i][j] = self.__cal_alpha(x, y)
                    betas[i][j] = self.__cal_beta(x, y)

        elif dimension  == 1:
            for i in range(3):
                x = ex_s[i]
                y = var_s[i]
                alphas[i]  = self.__cal_alpha(x, y)
                betas[i] = self.__cal_beta(x, y)
        return alphas, betas

    def __cal_alpha(self, ex, var):
        x = ex
        y = var
        return -(x*y + x**3 - x**2) / y

    def __cal_beta(self, ex, var):
        x = ex
        y = var
        return (x * (y + 1) - y + x**3 - 2 * x**2 ) / y

    def find_beta_sample(self, modain, i, j=0):
        if modain == "topography":
            # T_sample represents the probability of transitioning from vegetation type i to j
            alpha = self.topography_alphas[i][j]
            beta = self.topography_betas[i][j]
        elif modain == "vegetation":
            # V_sample represents the probability of transitioning from vegetation type i to j
            alpha = self.vegetation_alphas[i][j]
            beta = self.vegetation_betas[i][j]
        elif modain == "slope":
            # S_sample  represents the probability of following a certain local slope type i
            alpha = self.slope_alphas[i]
            beta = self.slope_betas[i]
        else:
            print("find_beta_sample have error !!!")
        sample = np.random.beta(alpha, beta)
        return sample


if __name__ ==  "__main__":
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

    transfer_matrix = Transfer_matrix(topography_exs, topography_vars, 
        vegetation_exs, vegetation_vars, slope_exs, slope_vars)
    sample = transfer_matrix.find_beta_sample("topography", 0, 0)
    print("sample = ", sample)

    

    Ti, Tj = 2, 2
    Vi, Vj = 2, 2
    Si = 2


    index = 0
    # states contains all cells states, each cell's state is like [0/1/2, 0/1/2, 0/1/2, 8848.0, index]
    cell_state_current = states[index]
    cell_state_7neighbors = find_state_7neighbors(states, index)

    # cell_state[0] use 0, 1, 2 represents topography:  lake, plain, hill 
    # cell_state[1] use 0, 1, 2 represents vegetation density: sparse, medium,dense
    # cell_state[2] use float value represents elevation
    # cell_state[3] use int represents the index of this cell
    fai = [0 for i in range(7)]
    for cell_state_neighbor in cell_state_7neighbors:
        fai[i] = combine_probilities(cell_state_current, cell_state_neighbor)
    fai = normalize_probilities(fai)



