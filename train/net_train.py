from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.dataloader import *
from utils.utils import *
from cnn_module.cnn_2d import ClassificationNet
import numpy as np


USE_GPU = 0
CUDA_NUM = 0
BATCH_SIZE = 1
EPOCH_TIMES = 5
CLASS_NUM = 5

WEIGHTS_NAME = 'data/weights/ClassificationNet' \
               + '_epoch_times_' + str(EPOCH_TIMES) \
               + '_batch_size_' + str(BATCH_SIZE)


if __name__ == "__main__":
    img_path = 'data/train/images'
    label_path = 'data/train/label'
    data_set = HgDataset(img_path, label_path)
    train_loader = data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    net = ClassificationNet(CLASS_NUM)
    if USE_GPU:
        net.cuda(CUDA_NUM)

    class_loss_layer = nn.CrossEntropyLoss()

    lr = 0.001
    optimizer = optim.SGD([param for param in net.parameters() if param.requires_grad is True], lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

    for epoch in range(EPOCH_TIMES):
        loss_train_epoch = 0

        for i, data in enumerate(train_loader):
            net.train()
            print("training batch#", i, ", epoch#", epoch)

            # data prepare
            img, label = data
            label = label_transform(label, CLASS_NUM)
            img, label = Variable(img), Variable(label)
            if USE_GPU:
                img, label = img.cuda(CUDA_NUM), label.cuda(CUDA_NUM)

            optimizer.zero_grad()
            class_out = net(img)
            print(class_out)
            print(label)
            class_loss = class_loss_layer(class_out, label)
            loss_train = class_loss
            loss_train.backward()
            scheduler.step()
            optimizer.step()








