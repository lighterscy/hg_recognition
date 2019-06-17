import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


class ClassificationNet(nn.Module):
    """
    hand gesture classificatioin module with 1-d cnn

    """
    def __init__(self, output_num):
        super(ClassificationNet, self).__init__()
        self.feature_extract_module = _FeatureExtract(BasicBlock)

        self.classification_module = nn.Sequential(nn.Linear(128*9, 100),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(100, output_num))

    def forward(self, wave):
        feature_out = self.feature_extract_module(wave)
        feature_flat = feature_out.reshape(np.shape(feature_out)[0], -1)
        final_out = self.classification_module(feature_flat)

        return final_out


class _FeatureExtract(nn.Module):
    """

    base feature extraction module
    input data is 2 * point_num signal, 2 includes real part and imag part of the signal

    """
    def __init__(self, block):
        super(_FeatureExtract, self).__init__()
        self.block = block
        self.layer1 = self._make_layer(2, 12, 16)
        self.layer2 = self._make_layer(12, 32, 8)
        self.layer3 = self._make_layer(32, 64, 4)
        self.layer4 = self._make_layer(64, 128, 2)
        self.avgpool = nn.AvgPool1d(2)

    def _make_layer(self, inplanes, outplanes, filter_num):
        layers = [self.block(inplanes, outplanes, filter_num)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, filter_num):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, outplanes, filter_num, stride=2)
        self.conv2 = nn.Conv1d(outplanes, outplanes, filter_num)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        return x


if __name__ == "__main__":
    net = ClassificationNet(5)
    print(net)
    input = Variable(torch.zeros(1, 2, 5000))
    out = net(input)
    print(np.shape(out))

    print('# generator parameters:', sum(param.numel() for param in net.parameters()))


