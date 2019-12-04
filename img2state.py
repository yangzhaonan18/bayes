import cv2
import math

class State():
    def __init__(self, T_path, V_path, E_path):
        self.H = 500
        self.W = 1000
        self.W_num = 32
        self.H_num = 19
        self.r = (self.H / (2 * self.H_num)) * (2 / math.sqrt(3)) + 2
 

        T_img = cv2.imread(T_path, cv2.IMREAD_COLOR)
        V_img = cv2.imread(V_path, cv2.IMREAD_COLOR)
        print("original size is = ", T_img.shape)
        self.T_img = cv2.resize(T_img, (self.W, self.H))
        self.V_img = cv2.resize(V_img, (self.W, self.H))
        print("after resize size is = ", self.T_img.shape)  # (518, 1033, 3)
 


    def points(self):
 
        print("radius = ", self.r) 

        print(self.T_img.shape)
        print(self.T_img[0][0])
        v = self.T_img[0][0][0]
        print(type(v))


        for y in range(self.W_num):
            pen_y = self.r * (math.sqrt(3) / 2 ) * y   + (self.r / 2) * math.sqrt(3)#  1.5 * r * y
            pen_x = 2 * self.r - self.r * 0.75 * math.pow(-1, y) # (r / 4) * math.sqrt(3) * math.pow(-1, y + 1) + (r / 4) * math.sqrt(3) + r
            pen_x_ = pen_x 
            for x  in range(self.H_num):
                # The first and the last share special color and size labels
                if (x == 0 and y == 0) or (x == self.H_num -1 and y == self.W_num - 1):
                    print("(int(pen_x_), int(pen_y))", (int(pen_x_), int(pen_y)))
                    color = (0, 0, 255)
                    cv2.circle(self.T_img, (int(pen_x_), int(pen_y)), radius=10, color=color, thickness=-1)  
                # type 0
                if self.T_img[int(pen_y)][int(pen_x_)][0] <  100:
                    color =  (0, 255, 255 )
                # type 2
                elif self.T_img[int(pen_y)][int(pen_x_)][0] >  200:
                    color =  (0, 200, 0 )
                # The other is type 1
                else: 
                    color =  (255, 0, 0 )
                    

                cv2.circle(self.T_img, (int(pen_x_), int(pen_y)), radius=2, color=color, thickness=2)
                pen_x_ += 3 * self.r # r * math.sqrt(3)

             

                # print("x, y = ", x, y)
                # print("pen_y pen_x_ =  ", int(pen_x_), int(pen_y))

        cv2.imshow("T_img", self.T_img)
        cv2.waitKey(0)    

        


    def T():
        pass

    def V():
        pass

    def E():
        pass












# cv2.imshow("a", T_img)
# cv2.waitKey(10)

if __name__ =="__main__":
    T_path = "./data_TVE/T.png"
    V_path = "./data_TVE/V.png"
    E_path = "./data_TVE/E.img"

    state = State(T_path, V_path, E_path)
    state.points()
    # T = state.T()




