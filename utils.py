
import numpy as np
import math
import cv2
import copy

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

    def find_beta_sample(self, geographic, i, j=0):
        if geographic == "topography":
            # T_sample represents the probability of transitioning from vegetation type i to j
            alpha = self.topography_alphas[i][j]
            beta = self.topography_betas[i][j]
        elif geographic == "vegetation":
            # V_sample represents the probability of transitioning from vegetation type i to j
            alpha = self.vegetation_alphas[i][j]
            beta = self.vegetation_betas[i][j]
        elif geographic == "slope":
            # S_sample  represents the probability of following a certain local slope type i
            alpha = self.slope_alphas[i]
            beta = self.slope_betas[i]
        else:
            print("find_beta_sample have error !!!")
        sample = np.random.beta(alpha, beta)
        return sample





class Build_State():
    def __init__(self, T_path, V_path, E_path):
        self.H = 500
        self.W = 1000
        self.H_num = 32
        self.W_num = 19
        self.r = self.H /self.H_num  + 1  # 16.625
        self.TVE_states = np.zeros((self.H_num, self.W_num, 3))

        T_img = cv2.imread(T_path, cv2.IMREAD_COLOR)
        V_img = cv2.imread(V_path, cv2.IMREAD_COLOR)
        self.T_img = cv2.resize(T_img, (self.W, self.H))
        self.V_img = cv2.resize(V_img, (self.W, self.H))



        self.__TV()

        print("self.r = ", self.r)
        print("original size is = ", T_img.shape)
        print("after resize size is = ", self.T_img.shape)  # (518, 1033, 3)



    def __TV(self):

        print("radius = ", self.r) 
        T_img_copy = copy.deepcopy(self.T_img)
        V_img_copy = copy.deepcopy(self.V_img)

        print(self.T_img.shape)
        print(self.T_img[0][0])
        v = self.T_img[0][0][0]
        print(type(v))
        
        # print(TVE_states)
        print(self.TVE_states.shape)
        for y in range(self.H_num):
            pen_y = self.r * (math.sqrt(3) / 2 ) * y   + (self.r / 2) * math.sqrt(3)#  1.5 * r * y
            pen_x = 2 * self.r - self.r * 0.75 * math.pow(-1, y) # (r / 4) * math.sqrt(3) * math.pow(-1, y + 1) + (r / 4) * math.sqrt(3) + r
            pen_x_ = pen_x 
            for x  in range(self.W_num):
                print(x, y)
                # The first and the last share special color and size labels
                if (x == 0 and y == 0) or (x == self.W_num -1 and y == self.H_num - 1):

                    print("(int(pen_x_), int(pen_y))", (int(pen_x_), int(pen_y)))
                    color = (0, 0, 255)
                    cv2.circle(T_img_copy, (int(pen_x_), int(pen_y)), radius=10, color=color, thickness=-1)
                    cv2.circle(V_img_copy, (int(pen_x_), int(pen_y)), radius=10, color=color, thickness=-1)  
                # #########
                # 
                # Use the information in topography(T.img) to determine the value of  TVE_states's topography values
                # 
                # ##########

                # type 0
                if self.T_img[int(pen_y)][int(pen_x_)][0] <  100:
                    self.TVE_states[y][x][0] = 0
                    T_color =  (0, 0, 255)
                # type 2
                elif self.T_img[int(pen_y)][int(pen_x_)][0] >  200:
                    self.TVE_states[y][x][0] = 2
                    T_color =  (0, 255, 0 )
                # The other is type 1
                else: 
                    self.TVE_states[y][x][0] = 1
                    T_color =  (255, 0, 0 )

                # #########
                # 
                # Use the information in vegetation(V.img) to determine the value of  TVE_states's vegetation values
                # 
                # ##########
                # type 0
                if self.V_img[int(pen_y)][int(pen_x_)][0] <  100:
                    self.TVE_states[y][x][1] = 0
                    V_color =  (0, 0, 255 )
                # type 2
                elif self.V_img[int(pen_y)][int(pen_x_)][0] >  200:
                    self.TVE_states[y][x][1] = 2
                    V_color =  (0, 255, 0 )
                # The other is type 1
                else: 
                    self.TVE_states[y][x][1] = 1
                    V_color =  (255, 0, 0 )

                cv2.circle(T_img_copy, (int(pen_x_), int(pen_y)), radius=2, color=T_color, thickness=2)
                cv2.circle(V_img_copy, (int(pen_x_), int(pen_y)), radius=2, color=V_color, thickness=2)
 






             
                pen_x_ += 3 * self.r # r * math.sqrt(3)
                # print("x, y = ", x, y)
                # print("pen_y pen_x_ =  ", int(pen_x_), int(pen_y))

        cv2.imshow("T_img_copy", T_img_copy)
        cv2.imshow("V_img_copy", V_img_copy)
        cv2.waitKey(0)   

        TVE_states = None 
        return TVE_states

        

    def E():
        pass













