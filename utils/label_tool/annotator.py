import os
from copy import deepcopy
import cv2
import utils.image_browser as ib


if __name__ == "__main__":
    file_dir = 'data/anno_test'
    ib.init(file_dir)

    json_path = os.path.join(file_dir, 'annotation.json')

