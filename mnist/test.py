import time
import copy
import numpy as np
import math

from options import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from models.classifier import Model
from data.mnist_loader import MNIST_Loader
from util.visualizer import Visualizer


def model_state_dict_parallel_convert(state_dict, mode):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    if mode == 'to_single':
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'to_parallel':
        for k, v in state_dict.items():
            name = 'module.' + k  # add 'module.' of DataParallel
            new_state_dict[name] = v
    elif mode == 'same':
        new_state_dict = state_dict
    else:
        raise Exception('mode = to_single / to_parallel')

    return new_state_dict


if __name__=='__main__':
    trainset = MNIST_Loader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)

    testset = MNIST_Loader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)
    print('#testing point clouds = %d' % len(testset))

    # create model, optionally load pre-trained model
    model = Model(opt)
    model.encoder.load_state_dict(
        model_state_dict_parallel_convert(torch.load(
            '/ssd/rotation-so-net/mnist/checkpoints/save/rot12/feature1024-batch64-RotAug-2/294_0.981000_net_gpu_2_encoder.pth', map_location='cpu'),
                                          mode='same'))
    model.classifier.load_state_dict(
        model_state_dict_parallel_convert(torch.load(
            '/ssd/rotation-so-net/mnist/checkpoints/save/rot12/feature1024-batch64-RotAug-2/294_0.981000_net_gpu_2_classifier.pth', map_location='cpu'),
                                          mode='same'))
    # model.encoder.load_state_dict(
    #     torch.load('/ssd/rotation-so-net/mnist/checkpoints/save/rot1/182_0.939000_net_gpu_1_encoder.pth'))
    # model.classifier.load_state_dict(
    #     torch.load('/ssd/rotation-so-net/mnist/checkpoints/save/rot1/182_0.939000_net_gpu_1_classifier.pth'))

    visualizer = Visualizer(opt)

    # test network
    batch_amount = 0
    model.test_loss.data.zero_()
    model.test_accuracy.data.zero_()

    for i, data in enumerate(testloader):
        B = data[0].size()[0]
        C = opt.classes

        input_pc, input_sn, input_label, input_node, input_node_knn_I = data

        model.set_input(input_pc, input_sn, input_label, input_node, input_node_knn_I)
        model.test_model()

        # calculate voted score/prediction
        batch_amount += B

        # accumulate loss
        model.test_loss += model.loss.detach() * B

        # accumulate accuracy
        _, predicted_idx = torch.max(model.score.detach(), dim=1, keepdim=False)
        correct_mask = torch.eq(predicted_idx, model.label).float()
        test_accuracy = torch.mean(correct_mask).cpu()
        model.test_accuracy += test_accuracy * B

    model.test_loss /= batch_amount
    model.test_accuracy /= batch_amount

    print('test sample number %d' % batch_amount)
    print('Loss %f, accuracy %f' % (model.test_loss.item(), model.test_accuracy.item()))


