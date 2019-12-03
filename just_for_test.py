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




######################################## 
import turtle

import math

W = 970
H = 500
W_num = 32
H_num = 19
 
r = int((500 / 33) * (2 / math.sqrt(3)))  # H / ( 2 * H_num)

 

turtle.speed(0)
turtle.setup(W, H, None, None)
turtle.bgpic('./data_TVE/T.png')
turtle.pencolor("red")
for y in range(H_num):
    turtle.pen_y = H / 2 - 1.5 * r * y - r * 2
    turtle.pen_x = -W / 2 - (r / 4) * math.sqrt(3) * math.pow(-1, y) + (r / 4) * math.sqrt(3) + r
    turtle.penup()
    turtle.goto(turtle.pen_x, turtle.pen_y)
    turtle.pendown()
    for x in range(W_num):
        turtle.circle(r, steps=6)
        turtle.penup()
        turtle.forward(r * math.sqrt(3))
        turtle.pendown()
        print(turtle.position())
        # if 

turtle.done()
