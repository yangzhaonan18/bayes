# import math

# pi = 3.1416
# print(math.sin(30 * (pi / 180)))


################################################################
# import random

# class Transfer_matrix:
# 	def __init__(self,):
# 		self.name  = "yzn"
# 		self.x = 108
# 		self.y = 2
# 		self.z = self.__cal(self.x, self.y)
	
# 	def add_one(self,):
# 		print("666")
	
# 	def __cal(self,x, y):
# 		return x + y  + random.randint(100, 100000)
# 	def p(self,):
# 		print(self.z)
	
		 



# if __name__ == "__main__":
# 	tm = Transfer_matrix()
# 	x = tm.add_one()
# 	tm.p()
# 	tm.p()
# 	tm.p()
# 	tm.p()
# 	tm.p()
# 	tm.p()
# 	tm.p()
# 	tm.p()
# #######################################


import numpy as np

X = np.arange(0, 5, 1)
Y = np.arange(0, 2, 1)
print(X)
X, Y = np.meshgrid(X, Y)
print(X)
print(Y)


R = np.sqrt(X**2 + Y**2)
print(R)