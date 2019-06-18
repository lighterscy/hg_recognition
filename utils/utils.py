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
    print(label)
    class_batch = torch.Tensor(label.shape[0], class_num)  # TODO
    class_tensor = torch.zeros(class_num)
    class_tensor[int(label[0])-1] = 1
    return class_tensor