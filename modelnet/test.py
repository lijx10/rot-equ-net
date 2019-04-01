import time
import copy
import numpy as np
import math

from options import Options
opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch

############################# automation for ModelNet10 / 40 configuration ####################
if opt.classes == 10:
    opt.dropout = opt.dropout + 0.1
############################# automation for ModelNet10 / 40 configuration ####################

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



def main():
    while opt.batch_size * opt.rot_equivariant_no * opt.input_pc_num > 8*12*1024:
        opt.batch_size = round(opt.batch_size / 2)
    print('batch_size %d ' % opt.batch_size)

    trainset = ModelNet_Shrec_Loader(opt.dataroot, 'train', opt)
    dataset_size = len(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                                              num_workers=opt.nThreads)

    testset = ModelNet_Shrec_Loader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=opt.nThreads)
    # create model, optionally load pre-trained model
    model_path = 'net_gpu_2'
    print(model_path)
    model = Model(opt)
    model.encoder.load_state_dict(
        model_state_dict_parallel_convert(torch.load(
            model_path+'_encoder.pth', map_location='cpu'),
            mode='same'))
    model.classifier.load_state_dict(
        model_state_dict_parallel_convert(torch.load(
            model_path+'_classifier.pth', map_location='cpu'),
            mode='same'))

    visualizer = Visualizer(opt)

    # test network
    batch_amount = 0
    model.test_loss.data.zero_()
    model.test_accuracy.data.zero_()

    per_class_correct = np.zeros(opt.classes)
    per_class_amount = np.zeros(opt.classes)
    per_class_acc = np.zeros(opt.classes)

    softmax = torch.nn.Softmax(dim=1).to(opt.device)

    voting_num = 12
    for i, data in enumerate(testloader):
        B = data[0].size()[0]
        C = opt.classes

        input_pc, input_sn, input_label, input_node, input_node_knn_I = data

        # perform voting
        score_sum = torch.zeros((B, C), dtype=torch.float32, device=opt.device, requires_grad=False)  # BxC
        loss_sum = torch.tensor([0], dtype=torch.float32, device=opt.device, requires_grad=False)
        for v in range(voting_num):
            if opt.rot_equivariant_mode == '2d':
                angle = (2 * math.pi / voting_num) * v
                rot_input_pc, rot_input_sn, rot_input_node = augmentation.rotate_point_cloud_with_normal_som_pytorch_batch(
                    input_pc,
                    input_sn,
                    input_node,
                    angle)
            elif opt.rot_equivariant_mode == '3d':
                rot_input_pc, rot_input_sn, rot_input_node = augmentation.rotate_point_cloud_with_normal_som_pytorch_batch_3d(
                    input_pc,
                    input_sn,
                    input_node)
            else:
                raise Exception('wrong mode.')


            model.set_input(rot_input_pc, rot_input_sn, input_label, rot_input_node, input_node_knn_I)
            model.test_model()

            # accumulate score
            score_sum += softmax(model.score.detach())
            # score_sum += model.score.detach()
            loss_sum += model.loss.detach()

        # calculate voted score/prediction
        batch_amount += B

        # accumulate loss
        model.test_loss += (loss_sum / voting_num) * B

        # accumulate accuracy
        _, predicted_idx = torch.max(score_sum, dim=1, keepdim=False)
        correct_mask = torch.eq(predicted_idx, model.label).float()
        test_accuracy = torch.mean(correct_mask).cpu()
        model.test_accuracy += test_accuracy * B

        # per class accuracy
        for b in range(model.label.size()[0]):  # tensor
            per_class_amount[model.label[b]] += 1
            if correct_mask[b] >= 0.9:
                per_class_correct[model.label[b]] += 1

    model.test_loss /= batch_amount
    model.test_accuracy /= batch_amount

    print('test sample number %d' % batch_amount)
    print('Loss %f, accuracy %f' % (model.test_loss.item(), model.test_accuracy.item()))

    # per class accuracy
    per_class_acc = per_class_correct / per_class_amount
    print('Per class accuracy: %f' % np.mean(per_class_acc))

    return model.test_accuracy.item(), np.mean(per_class_acc)


if __name__=='__main__':
    accuracy_sum = 0
    per_class_accuracy_sum = 0
    total_number = 10
    for trial in range(total_number):
        accuracy_i, per_class_acc_i = main()
        accuracy_sum += accuracy_i
        per_class_accuracy_sum += per_class_acc_i


    accuracy = accuracy_sum / total_number
    print('%d average accuracy: %f' % (total_number, accuracy))
    per_class_accuracy = per_class_accuracy_sum / total_number
    print('%d average per-class accuracy: %f' % (total_number, per_class_accuracy))

