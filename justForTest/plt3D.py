import gdal 
import numpy as np
import cv2
# import geos

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


geo = gdal.Open('../data_TVE/E.img') 
data = geo.ReadAsArray()
# print("data.shape = ", data.shape)  # data.shape =  (35, 67)
min_e = np.min(data)
max_e = np.max(data)
# print(min_e)  # 2398.6328
# print(max_e)  # 2474.1682
# print(max_e - min_e)  # 75.5354
b = np.array(255 * ((data - min_e)/(max_e - min_e))).astype(np.uint8)
g = b
r = b
img = cv2.merge([b, g, r])
img = cv2.resize(img, (1000, 500))  #  h 500  w 10000
# cv2.imshow('img', img)
# cv2.waitKey(0)


def show_img3D(img_cv):
	b, g, r = cv2.split(img_cv)
	h, w, c = img_cv.shape 

	fig = plt.figure()
	ax = Axes3D(fig)
	Y = np.arange(0, h, 1)  # 500
	X = np.arange(0, w, 1)  # 1000
	X, Y = np.meshgrid(X, Y)
	Z = np.array(b)
	# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

	ax.plot_surface(Y, X, Z, cmap='rainbow')
	plt.show()

if __name__ == '__main__':
	show_img3D(img)