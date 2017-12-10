import numpy as np

import numpy.matlib as matlib
a = np.zeros([4,3])



IMAGE_HEIGHT=a.shape[0]
IMAGE_WIDTH=a.shape[1]
center_x=IMAGE_WIDTH/2
center_y=IMAGE_HEIGHT/2
R = np.sqrt(center_x**2 + center_y**2)
mask_x = matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
mask_y = matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)
x1 = np.arange(IMAGE_WIDTH)
x_map = matlib.repmat(x1, IMAGE_HEIGHT, 1)
y1 = np.arange(IMAGE_HEIGHT)
y_map = matlib.repmat(y1, IMAGE_WIDTH, 1)
y_map = np.transpose(y_map)
Gauss_map = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2)
Gauss_map = np.exp(-0.5*Gauss_map/R)


print Gauss_map.shape
print Gauss_map