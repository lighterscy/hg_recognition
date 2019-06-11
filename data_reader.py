from scipy.io import loadmat
import numpy as np
import cv2
import os


"""
1 上 2 下 3 上下 4 下上 5 左右
"""


def mat2img(mat_path, save_path, name):
    mat = loadmat(mat_path)
    data = mat["SpecAll"]
    data = (data - np.min(data))/(np.max(data)-np.min(data))
    data *= 255
    cv2.imwrite(save_path+'/'+str(name)+'.jpg', data)


if __name__ == "__main__":
    counter = 1
    save_path = 'data/dataaft'
    mat_path = 'data/dataori'
    dir_files = os.listdir(mat_path)
    for i in dir_files:
        file_path = os.path.join(mat_path, i)

        dir_mat = os.listdir(file_path)
        for j in dir_mat:
            if j[-3:] == 'mat':
                single_path = os.path.join(file_path, j)
                mat2img(single_path, save_path, counter)
                counter += 1


