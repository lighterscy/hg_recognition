import random
import torch.utils.data as data
import torch


def generate_train_val_sampler(total_train_size, train_ratio):
    """
    in pytorch, use sampler class in dataloader to generater trainset and valset
    this function return two sampler, train_sampler and val_sampler

    :param total_train_size: int. trainset and valset is splited from the total trainset.
            this parameter is the size of the total trainset
    :param train_ratio: ratio of the train set size in total train size. e.g., if
            train_ratio is 0.8, will generate 4/5 trainset, 1/5 valset of total trainset
    :return:
    """
    random.seed(123)  # CAUTION!!!  this line is just for developing

    indices = list(range(total_train_size))
    random.shuffle(indices)
    train_indices = indices[:int(total_train_size * train_ratio)]
    val_indices = indices[int(total_train_size * train_ratio):]

    train_sampler = data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler


def label_transform(label, class_num):
    class_batch = torch.LongTensor(label.shape[0])  # 直接用torch.tensor默认为float, 但交叉熵默认需要输入long格式
    for i in range(label.shape[0]):
        class_batch[i] = label[i]
    return class_batch