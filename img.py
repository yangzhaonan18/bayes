import gdal 
# import geos

geo = gdal.Open('data/clip_USGS_NED_13_n40w112_IMG/clip_USGS_NED_13_n40w112_IMG.img') 
data = geo.ReadAsArray() 
# print(repr(data) )
# print(data)
print(data.shape)
 
h, w = data.shape
 


from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
# X = np.arange(-4, 4, 0.25)
X = np.arange(0, h, 1)
Y = np.arange(0, w, 1)
X, Y = np.meshgrid(X, Y)
print(X.shape)
data = data.T
Z = np.array(data)
print(Z.shape)
# R = np.sqrt(X**2 + Y**2)

# Z = np.sin(R)

 
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

plt.show()




 