from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.dataloader import HgDataset
from utils.utils import label_transform, generate_train_val_sampler
from cnn_module.cnn_2d import ClassificationNet
from cnn_module.detection_model import DetectionNet
import numpy as np

USE_GPU = 1
CUDA_NUM = 0
BATCH_SIZE = 8
EPOCH_TIMES = 500
CLASS_NUM = 5

WEIGHTS_NAME = 'data/weights/ClassificationNet' \
               + '_epoch_times_' + str(EPOCH_TIMES) \
               + '_batch_size_' + str(BATCH_SIZE)


if __name__ == "__main__":

    # read dataset
    img_path = 'data/train/images'
    label_path = 'data/train/label'
    data_set = HgDataset(img_path, label_path)

    # generate train & val
    train_sampler, val_sampler = generate_train_val_sampler(len(data_set), 0.8)
    train_loader = data.DataLoader(data_set, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
    val_loader = data.DataLoader(data_set, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2)

    # define net
    net = ClassificationNet(CLASS_NUM)
    # net = DetectionNet()
    if USE_GPU:
        net.cuda(CUDA_NUM)

    # define loss
    class_loss_layer = nn.CrossEntropyLoss()
    loss_train_list = []
    loss_val_list = []

    # define optimizer & scheduler
    lr = 0.001
    optimizer = optim.SGD([param for param in net.parameters() if param.requires_grad is True], lr, momentum=0.9)
    # optimizer = optim.Adam([param for param in net.parameters() if param.requires_grad if True], lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    for epoch in range(EPOCH_TIMES):
        loss_train_epoch = 0
        loss_val_epoch = 0

        '''
        training process
        '''
        net.train()
        for i, data in enumerate(train_loader):
            # print("training batch#", i, ", epoch#", epoch)

            # data prepare
            img, label = data
            label = label_transform(label)
            img, label = Variable(img), Variable(label)
            if USE_GPU:
                img, label = img.cuda(CUDA_NUM), label.cuda(CUDA_NUM)

            optimizer.zero_grad()
            class_out = net(img)
            # print(class_out, label)
            class_loss = class_loss_layer(class_out, label)
            loss_train = class_loss
            loss_train.backward()
            scheduler.step()
            optimizer.step()

            # print("batch loss:", loss_train.item())
            loss_train_epoch += loss_train.item()

        loss_train_epoch = loss_train_epoch / len(train_sampler) * BATCH_SIZE
        # loss_train_list.append(loss_train_epoch)
        # print("training loss is ", loss_train_epoch, "for epoch #", epoch)

        '''
        validation process
        '''
        net.eval()
        wrong = 0
        for data in val_loader:
            img, label = data
            label = label_transform(label)
            img, label = Variable(img), Variable(label)
            if USE_GPU:
                img, label = img.cuda(CUDA_NUM), label.cuda(CUDA_NUM)
            class_out = net(img)
            # print(class_out, label)
            for i in range(len(label)):
                if np.argmax(class_out[i].cpu().detach().numpy()) != int(label[i]):
                    wrong += 1
            class_loss = class_loss_layer(class_out, label)
            loss_val = class_loss

            # print("batch loss:", loss_val.item())
            loss_val_epoch += loss_val.item()

            loss_val_epoch = loss_val_epoch / len(val_sampler) * BATCH_SIZE
            # loss_val_list.append(loss_val_epoch)
        # print("validation loss is ", loss_val_epoch, "for epoch #", epoch)
        print(wrong/len(val_sampler))



