from scipy.io import loadmat
import numpy as np
import cv2


# mat = loadmat('data/1sb_data1.mat')
#
# data = mat["SpecAll"]
# data = (data - np.min(data))/(np.max(data)-np.min(data))

mat = loadmat('data/3.mat')
print(mat)
data = mat["f_t_data_abs"]
print(np.shape(data))
data = (data - np.min(data))/(np.max(data)-np.min(data))
cv2.imshow("data", data)
cv2.waitKey(0)


# mat2 = loadmat('data/1.mat')
# print(mat2)
# data = mat2["f_t_data_abs"]
#
# cv2.imshow("data", data)
# cv2.waitKey(0)


