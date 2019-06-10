import json
import cv2
import copy


class PatchSet(object):
    def __init__(self):
        self.frame_number = 0
        self.frame_dict = {}


def load_patch_set(json_path):
    try:
        with open(json_path, 'r') as f:
            json_string = f.read()
    except IOError:
        print('cannot open ' + json_path + ', create a new one')
        with open(json_path, 'w+') as f:
            json_string = f.read()

    try:
        patch_dict = json.loads(json_string)
    except ValueError:
        print(1111)


json_path = 'data/sample3.json'
load_patch_set(json_path)