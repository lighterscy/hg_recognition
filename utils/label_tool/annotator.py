import os
from copy import deepcopy
import cv2
import utils.label_tool.image_browser as ib


WINDOW_NAME = 'selector'
window_location = (100, 100)    # default window location

_QUIT_KEY = 27
_NEXT_KEY = ord('e')
_LAST_KEY = ord('q')


def _quit_func():
    exit()


def _next_func():
    ib.next_image()


def _previous_func():
    ib.previous_image()


func_map = {
    _QUIT_KEY: _quit_func(),
    _NEXT_KEY: _next_func(),
    _LAST_KEY: _previous_func()
}


if __name__ == "__main__":
    file_dir = 'data/anno_test'
    ib.init(file_dir)

    json_path = os.path.join(file_dir, 'annotation.json')

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, window_location[0], window_location[1])

    while True:

        key = cv2.waitKey(0)

        if str(key) in func_map.keys():
            func_map[key]()


