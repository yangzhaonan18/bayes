import gdal 
import numpy as np
import cv2
# import geos

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


geo = gdal.Open('../data_TVE/E.img') 
data = geo.ReadAsArray()

data = data.T
# print(repr(data) )
# print(data)
print("data.shape = ", data.shape)
 
h, w = data.shape  # 35 67 

fig = plt.figure()
ax = Axes3D(fig)

X = np.arange(0, w, 1)
Y = np.arange(0, h, 1)
print("X = ", X.shape)
X, Y = np.meshgrid(X, Y)
print("X", X.shape)
print(type(data))
Z = np.array(data)

print(Z.shape)



ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()


# print("data.shape = ", data.shape)
# min_e = np.min(data)
# max_e = np.max(data)
# print(min_e)
# print(max_e)
# print(max_e - min_e)

# b = np.array(255 * ((data - min_e)/(max_e - min_e))).astype(np.uint8)
# g = b
# r = b

# img = cv2.merge([b, g, r])
# print(img.shape)
# img = cv2.resize(img, (1000, 500))
# print(img.shape)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # print(img)
# print(data.shape)

# cv2.imshow('img', img)
# cv2.waitKey(0)




#  