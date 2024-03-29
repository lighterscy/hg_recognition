import os
import copy
import torch
import torch.utils.data as data
import json
import cv2


def transformImgToTensor(img):
    """

    :param img: numpy.ndarray  (generated by cv2.imread(), in format (H,W,C))
    :return: img:torch.Tensor (in format (C,H,W))
    """

    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    img = img.float() / 255
    return img


class PatchInFrame(object):
    """
    Base is the basic patch class
    used for dumping into json file
    """
    def __init__(self, type, pos):
        self.type = type    # ball number, -1 for negative sample. type: int
        self.pos = copy.copy(pos)    # left up corner. type: tuple

    def __str__(self):
        return 'class: %s, coord: %s' % (self.type, self.pos)

    def dump(self):
        return self.__dict__


class HgDataset(data.Dataset):
    def __init__(self, img_path, label_path):

        self.length = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
        self.img_path = img_path
        self.label_path = label_path

    def __getitem__(self, index):
        name = 'annotation.json'
        dict_path = os.path.join(self.label_path, name)
        f = open(dict_path, 'r')
        json_string = f.read()
        f.close()
        dict_all = json.loads(json_string)
        self.length = dict_all['frame_number']

        dict_all_change = {}
        for image_id, image_label in dict_all['frame_dict'].items():
            # print(image_id, image_label)
            dict_all_change[image_id] = image_label

        image_number = list(dict_all['frame_dict'].keys())[index]  # 单局编号
        label = dict_all_change[image_number]-1

        img = cv2.imread(os.path.join(self.img_path, image_number + '.jpg'))  # 对应图像
        label = torch.Tensor([label])
        img_tensor = transformImgToTensor(img)

        return img_tensor, label

    def __len__(self):
        return self.length


label_path = 'data/train/label'  # 标记路径
name = 'annotation.json'
path = os.path.join(label_path, name)

img_path = 'data/train/images'  # 图像路径

if __name__ == '__main__':

    trainset = HgDataset(img_path, label_path)

    for img, label in trainset:
        print(img)
        pass


