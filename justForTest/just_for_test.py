# import math

# pi = 3.1416
# print(math.sin(30 * (pi / 180)))


################################################################
# import random

# class Transfer_matrix:
#   def __init__(self,):
#       self.name  = "yzn"
#       self.x = 108
#       self.y = 2
#       self.z = self.__cal(self.x, self.y)
    
#   def add_one(self,):
#       print("666")
    
#   def __cal(self,x, y):
#       return x + y  + random.randint(100, 100000)
#   def p(self,):
#       print(self.z)
    
         



# if __name__ == "__main__":
#   tm = Transfer_matrix()
#   x = tm.add_one()
#   tm.p()
#   tm.p()
#   tm.p()
#   tm.p()
#   tm.p()
#   tm.p()
#   tm.p()
#   tm.p()
# #######################################


# import numpy as np

# X = np.arange(0, 5, 1)
# Y = np.arange(0, 2, 1)
# print(X)
# X, Y = np.meshgrid(X, Y)
# print(X)
# print(Y)


# R = np.sqrt(X**2 + Y**2)
# print(R)


########################################
# x_num = 
# a = 1
# for i in range(1):
#     for j in range(1 ,   , a):
#         if j % 3 == 1 :
#             print(j)
################

# import turtle
# turtle.bgpic('./data_TVE/T.png')
# turtle.setup(1000, 1500, 200, 200)
# # turtle.penup()
# # turtle.pendown()
# turtle.pensize(5)
# turtle.pencolor("black")
# turtle.fd(80)
# for i in range(5):
#     turtle.left(60)
#     turtle.fd(80)
# turtle.done()




# ######################################## 
# import turtle

# import math
# import cv2


# H = 500
# W_num = 32
# H_num = 19
 
# r = (H / (2 * H_num)) * (2 / math.sqrt(3)) + 2
# print("radius = ", r) 

# T_img = cv2.imread('./data_TVE/T.png')
# print(T_img.shape)
# print(T_img[0][0])
# v = T_img[0][0][0]
# print(type(v))
# print(T_img[0][0][0])

# for y in range(W_num):
#     pen_y = r * (math.sqrt(3) / 2 ) * y   + (r / 2) * math.sqrt(3)#  1.5 * r * y
#     pen_x = 2 * r - r * 0.75 * math.pow(-1, y) # (r / 4) * math.sqrt(3) * math.pow(-1, y + 1) + (r / 4) * math.sqrt(3) + r
#     pen_x_ = pen_x 
#     for x  in range(H_num):
#         # The first and the last share special color and size labels
#         if (x == 0 and y == 0) or (x == H_num -1 and y == W_num - 1):
#             print("(int(pen_x_), int(pen_y))", (int(pen_x_), int(pen_y)))
#             color = (0, 0, 255)
#             cv2.circle(T_img, (int(pen_x_), int(pen_y)), radius=10, color=color, thickness=-1)  
#         # type 0
#         if T_img[int(pen_y)][int(pen_x_)][0] <  100:
#             color =  (0, 255, 255 )
#         # type 2
#         elif T_img[int(pen_y)][int(pen_x_)][0] >  200:
#             color =  (0, 200, 0 )
#         # The other is type 1
#         else: 
#             color =  (255, 0, 0 )
            

#         cv2.circle(T_img, (int(pen_x_), int(pen_y)), radius=2, color=color, thickness=2)
#         pen_x_ += 3 * r # r * math.sqrt(3)

     

#         # print("x, y = ", x, y)
#         # print("pen_y pen_x_ =  ", int(pen_x_), int(pen_y))

# cv2.imshow("T_img", T_img)
# cv2.waitKey(0)



#################################################


# for y in range(16):
#     pen_y = 1.5 * r * y + r *2
#     pen_x = (r / 4) * math.sqrt(3) * math.pow(-1, y + 1) + (r / 4) * math.sqrt(3) + r
#     pen_x_ = pen_x 
#     for x  in range(19):
#         cv2.circle(T_img, (int(pen_x_), int(pen_y)), radius=2, color=(255, 0, 0), thickness=2)
#         pen_x_ += r * math.sqrt(3)

# #         print("x, y = ", x, y)
# #         print("pen_y pen_x_ =  ", int(pen_x_), int(pen_y))

# # cv2.imshow("T_img", T_img)
# cv2.waitKey(0)

# a = "asdf"
# b = a.copy()  #  has no attribute 'copy'
# b = "a"
# print(a)






# #################


# W = 970
# H = 500
# W_num = 32
# H_num = 19
 
# r = int((500 / 33) * (2 / math.sqrt(3)))  # H / ( 2 * H_num)

# turtle.speed(0)
# turtle.setup(W, H, None, None)
# turtle.bgpic('./data_TVE/T.png')
# turtle.pencolor("red")
# for y in range(H_num):
#     turtle.pen_y = H / 2 - 1.5 * r * y - r * 2
#     turtle.pen_x = -W / 2 - (r / 4) * math.sqrt(3) * math.pow(-1, y ) + (r / 4) * math.sqrt(3) + r
#     turtle.penup()
#     turtle.goto(turtle.pen_x, turtle.pen_y)
#     turtle.pendown()
#     for x in range(W_num):
#         turtle.circle(r, steps=6)
#         turtle.penup()
#         turtle.forward(r * math.sqrt(3))
#         turtle.pendown()
#         # print(turtle.position())
#         # imgx = turtle.position()[0] +  W / 2
#         # imgy = turtle.position()[1]
    
#         # if l

# turtle.done()






# import cv2
# import numpy as np


# b = np.random.randint(0, 255, (5, 5), dtype=np.uint8)
# g = b
# r = b
# print(type(b))

# img = cv2.merge([b, g, r])
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cv2.imshow('img', img)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)


# cv2.destroyWindow('test')
# print(img)
# print(gray)




