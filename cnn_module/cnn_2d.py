import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class ClassificationNet(nn.Module):
    """
    hand gesture classificatioin module with 2-d cnn

    """
    def __init__(self, output_num=5):
        super(ClassificationNet, self).__init__()

        self.feature_extract_module = _FeatureExtract(BasicBlock)

        self.classification_module = nn.Sequential(nn.Linear(64*12*16, 100),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(100, output_num))
        # TODO: 修改FC 参数太多

    def forward(self, x):
        feature_out = self.feature_extract_module(x)
        # print(np.shape(feature_out))
        feature_flat = feature_out.reshape(np.shape(feature_out)[0], -1)
        final_out = self.classification_module(feature_flat)

        return final_out


class _FeatureExtract(nn.Module):
    """

    base feature extraction module
    input data is 100*100 stft img of signal

    """
    def __init__(self, block):
        super(_FeatureExtract, self).__init__()
        self.block = block
        self.layer1 = self._make_layer(3, 8, 3, 1)
        self.layer2 = self._make_layer(8, 16, 3, 2)
        self.layer3 = self._make_layer(16, 32, 3, 2)
        self.layer4 = self._make_layer(32, 64, 3, 2)

    def _make_layer(self, in_channels, out_channels, filter_size, stride):
        layers = [self.block(in_channels, out_channels, filter_size, stride)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, filter_size, stride=1)
        # self.conv2 = nn.Conv2d(out_channels, out_channels, filter_size, stride=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, filter_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # identity = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


if __name__ == "__main__":
    net = ClassificationNet(5)
    input = Variable(torch.zeros(1, 3, 128, 155))
    out = net(input)
    print(np.shape(out))
    print(type(out))
    print('# generator parameters:', sum(param.numel() for param in net.parameters()))

