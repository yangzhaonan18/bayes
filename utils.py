
import numpy as np
import math
import cv2
import copy
import gdal

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# print(type(alpha_s))
class Transfer_matrix():
    def __init__(self,topography_exs, topography_vars, vegetation_exs, vegetation_vars,
        slope_exs, slope_vars):
        # calculate alpha and beta from expectation and variance
        # topography, vegetation, and slope
        self.topography_alphas, self.topography_betas  = self.__calculate_alphas_betas(topography_exs, topography_vars)
        self.vegetation_alphas, self.vegetation_betas  = self.__calculate_alphas_betas(vegetation_exs, vegetation_vars)
        self.slope_alphas, self.slope_betas  = self.__calculate_alphas_betas(slope_exs, slope_vars)
        
        self.alpha_21s = self.__993_to_21(self.topography_alphas, self.vegetation_alphas, self.slope_alphas)
        self.beta_21s = self.__993_to_21(self.topography_betas, self.vegetation_betas, self.slope_betas)

        self.T_transition_matrix  = [[0 for i in range(3)] for i in range(3)]
        self.V_transition_matrix  = [[0 for i in range(3)] for i in range(3)]
        self.S_transition_matrix  = [[0 for i in range(3)] for i in range(3)]
        self.TVS_transition_matrix = [0 for i in range(21)]
        print("\n\n###     Alpha_s list below :  ###")
        print("\ntopography_alphas = \n", self.topography_alphas)
        print("\nvegetation_alphas = \n", self.vegetation_alphas)
        print("\nslope_alphas = \n", self.slope_alphas)

        print("\n\n###     Beta_s list below :  ###")
        print("\ntopography_betas = \n", self.topography_betas)
        print("\nvegetation_betas = \n", self.vegetation_betas)
        print("\nslope_betas = \n", self.slope_betas)
        print("\n")


    def  ___normalize_probilities(self, probability_list):
        # Normalize the probability values, make sure the sum of the probabilities is 1
        length = len(probability_list)
        probability_list_new = [0 for i in range(length)]
        total = 0
        for i in range(length):
            total += probability_list[i]
        for i in range(length):
            probability_list_new[i] = probability_list[i] / total 
        return probability_list_new




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


    def find_beta_sample_backup(self, geographic, i, j=0):
        sample_probilites_3s = [0, 0, 0]
        if geographic == "topography":
            # T_sample represents the probability of transitioning from vegetation type i to j
            for k in range(3):
                alpha = self.topography_alphas[i][k]
                beta = self.topography_betas[i][k]
                sample_probilites_3s[k] = np.random.beta(alpha, beta)
        elif geographic == "vegetation":
            # V_sample represents the probability of transitioning from vegetation type i to j
            for k in range(3):
                alpha = self.vegetation_alphas[i][k]
                beta = self.vegetation_alphas[i][k]
                sample_probilites_3s[k] = np.random.beta(alpha, beta)            
        elif geographic == "slope":
            # S_sample  represents the probability of following a certain local slope type i
            for k in range(3):
                alpha = self.slope_alphas[k]
                beta = self.slope_betas[k]
                sample_probilites_3s[k] = np.random.beta(alpha, beta)   
        # print("sample_probilites_3s = ", sample_probilites_3s)
        sample_probilites_3s_normalized = self.___normalize_probilities(sample_probilites_3s)
        # print(sample_probilites_3s_normalized)
        if geographic == "topography" or  geographic == "vegetation":
            return sample_probilites_3s_normalized[j]
        elif geographic == "slope":
            return sample_probilites_3s_normalized[i]
     

    def theta_sample_21s(self):
        for i in range(3):
            for j in range(3):
                self.T_alpha = self.topography_alphas[i][j]
                self.T_beta = self.topography_betas[i][j]
                self.T_transition_matrix[i][j] =  np.random.beta(self.T_alpha, self.T_beta)
                self.T_transition_matrix[i] = self.___normalize_probilities(self.T_transition_matrix[i])

        for i in range(3):
            for j in range(3):
                self.V_alpha  = self.vegetation_alphas[i][j]
                self.V_beta = self.vegetation_alphas[i][j]
                self.V_transition_matrix[i][j] =  np.random.beta(self.V_alpha , self.V_beta)
                self.V_transition_matrix[i] = self.___normalize_probilities(self.V_transition_matrix[i])


        for i in range(3):
            self.S_alpha = self.slope_alphas[i]
            self.S_beta = self.slope_alphas[i]
            self.S_transition_matrix[i] = np.random.beta(self.S_alpha, self.S_beta)  
        self.S_transition_matrix = self.___normalize_probilities(self.S_transition_matrix)

        self.TVS_transition_matrix = self.__993_to_21(self.T_transition_matrix, self.V_transition_matrix, self.S_transition_matrix)
        return self.TVS_transition_matrix

    def __993_to_21(self, A, B, C):
        ABC = [0 for i in range(21)]

        for i in range(3):
            ABC[18 + i] = C[i]
            for j in range(3):
                ABC[3 * i + j] = A[i][j]
                ABC[9 + 3 * i  + j] = B[i][j]
        return ABC

 





    def find_theta_sample(self, geographic, i, j=0):
        self.theta_sample_21s()
        if geographic == "topography":
            # print("self.T_transition_matrix[i][i] " ,self.T_transition_matrix[i][i])
            return self.T_transition_matrix[i][i] 
        elif geographic == "vegetation":
            # pirnt( self.V_transition_matrix[i][i])
            return self.V_transition_matrix[i][i]
        elif geographic == "slope":
            # print(self.S_transition_matrix[i])
            return self.S_transition_matrix[i]
        else:
            print("find_beta_sample function input error !")




    def change_beta_sample(geographic, new_value, index_ri, index_cj=-1):
        if geographic == "topography" :
            self.T_transition_matrix[index_ri][index_cj] = new_value
        elif geographic == "vegetation" :
            self.V_transition_matrix[index_ri][index_cj] = new_value
        elif geographic == "slope" :
            self.S_transition_matrix[index_ri] = new_value


     








class Build_State():
    """
    use three image to calculate every cells's states
    """
    def __init__(self, T_path, V_path, E_path):
        self.H = 500
        self.W = 1000
        self.H_num = 32
        self.W_num = 19
        self.r = self.H /self.H_num  + 1  # 16.625
        self.TVE_states_ij = np.zeros((self.H_num, self.W_num, 3))
        self.TVE_states_index = np.zeros((self.H_num * self.W_num, 3))

        T_img = cv2.imread(T_path, cv2.IMREAD_COLOR)
        V_img = cv2.imread(V_path, cv2.IMREAD_COLOR)
        self.T_img = cv2.resize(T_img, (self.W, self.H))
        self.V_img = cv2.resize(V_img, (self.W, self.H))
        self.E_img = np.zeros((self.W,self.H,3), np.uint8).fill(0)
        # cv2.imwrite("data_TVE/T_aNewSave.png", self.T_img)
        # cv2.imwrite("data_TVE/V_aNewSave.png", self.V_img)

        self.__resize_E_img(E_path)
        self.__calculate_TVE()

        # print("self.r = ", self.r)
        # print("original size is = ", T_img.shape)
        # print("after resize size is = ", self.T_img.shape)  # (518, 1033, 3)


    def __resize_E_img(self, E_path, readFromUSGSImg = True):
        readFromUSGSImg = False
        if readFromUSGSImg:  # E_original.img
            geo = gdal.Open(E_path) 
            data = geo.ReadAsArray()
            # print("data.shape = ", data.shape)  # data.shape =  (35, 67)
            min_e = np.min(data)  # 2398.6328
            max_e = np.max(data)  # 2474.1682
            # print(max_e - min_e)  # 75.5354
            b = np.array(75.5354 * ((data - min_e)/(max_e - min_e))).astype(np.uint8)
            g = b
            r = b
            img = cv2.merge([b, g, r])
            self.E_img = cv2.resize(img, (self.W, self.H))  #  h 500  w 10000
            cv2.imwrite("data_TVE/E_aNewSave.png", self.E_img)
        else:
            self.E_img = cv2.imread("data_TVE/E.png")
        # cv2.imshow('img', img)
        # cv2.waitKey(0)   
        

    def __calculate_TVE(self):
        print("radius = ", self.r) 
        self.T_img_copy = copy.deepcopy(self.T_img)
        self.V_img_copy = copy.deepcopy(self.V_img)
        v = self.T_img[0][0][0]
        for y in range(self.H_num):
            pen_y = self.r * (math.sqrt(3) / 2 ) * y   + (self.r / 2) * math.sqrt(3)#  1.5 * r * y
            pen_x = 2 * self.r - self.r * 0.75 * math.pow(-1, y) # (r / 4) * math.sqrt(3) * math.pow(-1, y + 1) + (r / 4) * math.sqrt(3) + r
            pen_x_ = pen_x 
            for x  in range(self.W_num):
                # print(x, y)
                # The first and the last share special color and size labels
                if (x == 0 and y == 0) or (x == self.W_num -1 and y == self.H_num - 1):

                    # print("(int(pen_x_), int(pen_y))", (int(pen_x_), int(pen_y)))
                    color = (0, 0, 255)
                    cv2.circle(self.T_img_copy, (int(pen_x_), int(pen_y)), radius=10, color=color, thickness=-1)
                    cv2.circle(self.V_img_copy, (int(pen_x_), int(pen_y)), radius=10, color=color, thickness=-1)  
                # #########
                # 
                # Use the information in topography(T.img) to determine the type of  TVE_states's topography type
                # 
                # ##########

                # type 0
                if self.T_img[int(pen_y)][int(pen_x_)][0] <  100:
                    self.TVE_states_ij[y][x][0] = 0
                    T_color =  (0, 0, 255)
                # type 2
                elif self.T_img[int(pen_y)][int(pen_x_)][0] >  200:
                    self.TVE_states_ij[y][x][0] = 2
                    T_color =  (0, 255, 0 )
                # The other is type 1
                else: 
                    self.TVE_states_ij[y][x][0] = 1
                    T_color =  (255, 0, 0 )

                # #########
                # 
                # Use the information in vegetation(V.img) to determine the type of  TVE_states's vegetation type
                # 
                # ##########
                # type 0
                if self.V_img[int(pen_y)][int(pen_x_)][0] <  100:
                    self.TVE_states_ij[y][x][1] = 0
                    V_color =  (0, 0, 255 )
                # type 2
                elif self.V_img[int(pen_y)][int(pen_x_)][0] >  200:
                    self.TVE_states_ij[y][x][1] = 2
                    V_color =  (0, 255, 0 )
                # The other is type 1
                else: 
                    self.TVE_states_ij[y][x][1] = 1
                    V_color =  (255, 0, 0 )
                # #########
                # 
                # Use the information in elevation(E.img) to determine the value of TVE_states's elevation value
                # 
                # ##########
                # type 0
                self.TVE_states_ij[y][x][2] = self.E_img[int(pen_y)][int(pen_x_)][0]
                cv2.circle(self.T_img_copy, (int(pen_x_), int(pen_y)), radius=2, color=T_color, thickness=2)

                cv2.circle(self.V_img_copy, (int(pen_x_), int(pen_y)), radius=2, color=V_color, thickness=2)
                cv2.putText(self.T_img_copy, str(y*self.W_num + x), (int(pen_x_), int(pen_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 170, 50), 2)

                pen_x_ += 3 * self.r # r * math.sqrt(3)
                # print("x, y = ", x, y)
                # print("pen_y pen_x_ =  ", int(pen_x_), int(pen_y))
        self.TVE_states_index = self.TVE_states_ij.reshape((self.H_num * self.W_num, 3))
 
    

    #   show value of elevation 
    # use 0~255 in image  to replace 0 to 75.5 m(relative elevation)
    def show_E_img3D(self):
        b, g, r = cv2.split(self.E_img)
        h, w, c = self.E_img.shape 

        fig = plt.figure()
        ax = Axes3D(fig)
        Y = np.arange(0, h, 1)  # 500
        X = np.arange(0, w, 1)  # 1000
        X, Y = np.meshgrid(X, Y)
        Z = np.array(b)
        # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
        ax.plot_surface(Y, X, Z, cmap='rainbow')
        plt.show()


    #  topography, vegetation. each have three types
    def show_TV_img2D(self):
        cv2.imshow("T_img_copy", self.T_img_copy)
        cv2.imshow("V_img_copy", self.V_img_copy)
        # cv2.waitKey(5000)  
        k = cv2.waitKey(0) 
        if k ==27:   
           cv2.destroyAllWindows() 







  






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






def show_probility_img3D(img):
    img = cv2.resize(img,(1000, 500))

    b, _, _ = cv2.split(img)
    h, w, _ = img.shape 
    max_p = np.max(img)
    min_p = np.min(img)
    b = np.array(255 * (b - min_p)/(max_p - min_p)).astype(np.uint8)

    fig = plt.figure()
    ax = Axes3D(fig)
    Y = np.arange(0, h, 1)  # 500
    X = np.arange(0, w, 1)  # 1000
    X, Y = np.meshgrid(X, Y)
    Z = np.array(b)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.plot_surface(Y, X, Z, cmap='rainbow')
    plt.show()        













