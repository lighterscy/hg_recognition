import os
import cv2

current_image = None
frame_id = None

_dir_path = None
_image_list = None
_image_index = 0


def init(dir_path):
    global _dir_path, _image_list
    _dir_path = dir_path
    file_list = os.listdir(_dir_path)
    _image_list = filter(lambda x: x.endswith('jpg'), file_list)
    _image_list = list(_image_list)
    _image_list.sort()
    _get_image()


def _get_image():
    global current_image, frame_id
    image_name = _image_list[_image_index]
    image_path = os.path.join(_dir_path, image_name)
    current_image = cv2.imread(image_path)
    frame_id = image_name[-8: -4]
    print(frame_id)


def next_image():
    global _image_index
    if _image_index < len(_image_list) - 1:
        _image_index += 1
        _get_image()
    else:
        print("It's the last one.")


def previous_image():
    global _image_index
    if _image_index > 0:
        _image_index -= 1
        _get_image()
    else:
        print("It's the first one.")


def delete_image():
    global _image_index
    image_name = _image_list[_image_index]
    image_path = os.path.join(_dir_path, image_name)
    _image_list.remove(image_name)
    os.remove(image_path)
    print(image_name, ' is deleted from image browser and filefolder')


