import timeit
import math

import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import numpy as np
# from scripts.img_proc.cnn_model.dataloader import *


class DetectionNet(nn.Module):
    """
    one-stage detection model

    """
    def __init__(self):
        super(DetectionNet, self).__init__()

        # resnet pretrained weights. Set pretrained=True when first training. Set pretrained=False when deploy.
        self.feature_extract_module = _resnet_features(pretrained=False)

        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.classification_module = nn.Sequential(nn.Linear(128*16*20, 128),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(128, 5))

    def forward(self, img):
        features = self.feature_extract_module(img)
        features = self.relu(self.bn1(self.conv1(features)))
        # print(np.shape(features))
        feature_flat = features.reshape(np.shape(features)[0], -1)
        class_tensor = self.classification_module(feature_flat)

        return class_tensor


class _ResNetFeatures(torchvision.models.resnet.ResNet):
    def delete_unused_layers(self):
        del self.avgpool
        del self.fc
        del self.layer4
        # del self.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        return x


def _resnet_features(pretrained=True, **kwargs):
    model = _ResNetFeatures(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torchvision.models.resnet.model_zoo.load_url(torchvision.models.resnet.model_urls['resnet18']))
    model.delete_unused_layers()
    return model


if __name__ == "__main__":
    # test the inference time of designed model
    detection_net = DetectionNet()
    detection_net.cuda(0)
    detection_net.eval()
    input = Variable(torch.zeros(1, 3, 128, 155)).cuda(0)
    # input = Variable(torch.cuda.FloatTensor(1, 3, 1080, 1440))
    # print input
    for time in range(1):
        torch.cuda.synchronize()
        tic = timeit.default_timer()
        class_tensor = detection_net(input)
        print(np.shape(class_tensor))
        # torch.cuda.synchronize()
        toc = timeit.default_timer()
        print("detection time: ", (toc - tic) * 1000, "ms")
    print('# generator parameters:', sum(param.numel() for param in detection_net.parameters()))


