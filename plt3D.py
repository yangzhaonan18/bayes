import gdal 
import numpy as np

# import geos

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


geo = gdal.Open('data_TVE/E.img') 
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
X, Y = np.meshgrid(X, Y)
Z = np.array(data)

print(Z.shape)



ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()




 