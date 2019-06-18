import os
from copy import deepcopy
import cv2
import utils.label_tool.image_browser as ib
import utils.label_tool.patch as pth


WINDOW_NAME = 'annotator'
window_location = (100, 100)    # default window location

_QUIT_KEY = 27
_NEXT_KEY = ord('e')
_LAST_KEY = ord('q')

_patch_set = None

_class_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '`': -1}


def imshow():
    image = ib.current_image.copy()
    if ib.frame_id in _patch_set.frame_dict.keys():
        cv2.putText(image, 'type: '+str(_patch_set.frame_dict[ib.frame_id]), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 5)
    cv2.putText(image, ib.frame_id, (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imshow(WINDOW_NAME, image)


def _quit_func():
    _patch_set.save_to_file(json_path)
    exit()


def _next_func():
    ib.next_image()


def _previous_func():
    ib.previous_image()


func_map = {
    _QUIT_KEY: _quit_func,
    _NEXT_KEY: _next_func,
    _LAST_KEY: _previous_func
}


def new_frame(key):
    patch = pth.Patch(ib.frame_id, int(key))
    _patch_set.add(patch)


if __name__ == "__main__":
    file_dir = 'data/train/images'
    ib.init(file_dir)

    json_path = os.path.join(file_dir, 'annotation.json')
    _patch_set = pth.load_patch_set(json_path)

    cv2.namedWindow(WINDOW_NAME)
    cv2.moveWindow(WINDOW_NAME, window_location[0], window_location[1])

    while True:
        imshow()
        key = cv2.waitKey(0)
        if chr(key) in _class_map.keys():
            new_frame(chr(key))

        elif key in func_map.keys():
            func_map[key]()


