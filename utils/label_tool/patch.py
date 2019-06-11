import json
import cv2
import copy


class Patch(object):
    def __init__(self, frame_id, type):
        self.frame_id = frame_id
        self.type = type
        self.dict = {self.frame_id: self.type}


class PatchSet(object):
    def __init__(self):
        self.frame_number = 0
        self.frame_dict = dict()

    def save_to_file(self, full_path):
        json_str = json.dumps(self.__dict__, indent=4)  # TODO dump changed
        f = open(full_path, 'w')
        f.write(json_str)
        f.close()

    def add(self, patch):
        if patch.frame_id not in self.frame_dict.keys():
            self.frame_number += 1
        self.frame_dict.update(patch.dict)

    def pop(self, frame_id):
        self.frame_dict.pop(frame_id)
        self.frame_number -= 1


def load_patch_set(json_path):
    try:
        with open(json_path, 'r') as f:
            json_string = f.read()
    except IOError:
        print('cannot open ' + json_path + ', create a new one')
        with open(json_path, 'w+') as f:
            json_string = f.read()
    patch_set = PatchSet()

    try:
        patch_dict = json.loads(json_string)
        for frame_id, frame_type in patch_dict['frame_dict'].items():
            patch = Patch(frame_id, frame_type)
            patch_set.add(patch)
    except ValueError:
        pass
    return patch_set


if __name__ == '__main__':
    test_path = 'data/anno_test/sample3.json'
    load_patch_set(test_path)
