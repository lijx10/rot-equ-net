import time
import copy
import numpy as np
import math

from options import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch
opt.batch_size = 1

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
from data import augmentation


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
            '/ssd/rotation-so-net/mnist/checkpoints/save/rot4/239_0.973000_net_gpu_0_encoder.pth', map_location='cpu'),
                                          mode='same'))
    model.classifier.load_state_dict(
        model_state_dict_parallel_convert(torch.load(
            '/ssd/rotation-so-net/mnist/checkpoints/save/rot4/239_0.973000_net_gpu_0_classifier.pth', map_location='cpu'),
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

        V_num = opt.rot_equivariant_no * 2
        for v in range(0, V_num):
            rotation_angle = v / V_num * 2
            rot_input_pc, rot_input_node = augmentation.rotate_pc_with_som_pytorch_batch(input_pc, input_node, rotation_angle)

            model.set_input(rot_input_pc, input_sn, input_label, rot_input_node, input_node_knn_I)
            model.test_model()

            # visualize

        break



