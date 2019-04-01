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
from data.modelnet_shrec_loader import ModelNet_Shrec_Loader
from data import augmentation
from util.visualizer import Visualizer


if __name__=='__main__':
    trainset = ModelNet_Shrec_Loader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.nThreads)
    print('#training point clouds = %d' % len(trainset))

    testset = ModelNet_Shrec_Loader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.nThreads)

    # create model, optionally load pre-trained model
    model = Model(opt)
    model.encoder.load_state_dict(
        torch.load('net_encoder.pth'))
    model.classifier.load_state_dict(
        torch.load('net_classifier.pth'))

    ############################# automation for ModelNet10 / 40 configuration ####################
    if opt.classes == 10:
        opt.dropout = opt.dropout + 0.1
    ############################# automation for ModelNet10 / 40 configuration ####################

    visualizer = Visualizer(opt)


    # test network
    model.test_loss.data.zero_()
    model.test_accuracy.data.zero_()

    softmax = torch.nn.Softmax(dim=1).to(opt.device)

    voting_num = 12
    C = opt.classes
    score = torch.zeros((len(testset), C, voting_num), dtype=torch.float32, device=opt.device, requires_grad=False)  # TotalxCxV
    label = torch.zeros((len(testset)), dtype=torch.int64, device=opt.device, requires_grad=False)  # Total
    loss_sum = torch.tensor([0], dtype=torch.float32, device=opt.device, requires_grad=False)
    for v in range(voting_num):
        angle = (2 * math.pi / voting_num) * v
        batch_amount = 0
        for i, data in enumerate(testloader):
            B = data[0].size()[0]

            input_pc, input_sn, input_label, input_node, input_node_knn_I = data
            rot_input_pc, rot_input_sn, rot_input_node = augmentation.rotate_point_cloud_with_normal_som_pytorch_batch(
                input_pc,
                input_sn,
                input_node,
                angle)
            model.set_input(rot_input_pc, rot_input_sn, input_label, rot_input_node, input_node_knn_I)
            model.test_model()

            # accumulate score
            score[batch_amount:batch_amount+B, :, v] = softmax(model.score.detach())  # BxC
            loss_sum += model.loss.detach() * B

            # accumulate label
            label[batch_amount:batch_amount+B] = model.label  # B

            batch_amount += B

    # average loss
    model.test_loss = loss_sum / voting_num / len(testset)

    # average accuracy
    score = torch.mean(score, dim=2, keepdim=False)  # TotalxC
    _, predicted_idx = torch.max(score, dim=1, keepdim=False)
    correct_mask = torch.eq(predicted_idx, label).float()
    model.test_accuracy = torch.mean(correct_mask).cpu()

    print('test sample number %d' % batch_amount)
    print('Loss %f, accuracy %f' % (model.test_loss.item(), model.test_accuracy.item()))
