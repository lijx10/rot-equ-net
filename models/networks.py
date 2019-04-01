import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
import math
import torch.utils.model_zoo as model_zoo
import time

from util import som
from . import operations
from .layers import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import index_max

class Transformer(nn.Module):
    def __init__(self, opt):
        super(Transformer, self).__init__()
        self.opt = opt

        self.first_pointnet = PointNet(3, (32, 64, 128), activation=self.opt.activation, normalization=self.opt.normalization,
                                       momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        self.second_pointnet = PointNet(128+128, (256, 256), activation=self.opt.activation, normalization=self.opt.normalization,
                                        momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        # regressor to get a sin(theta)
        self.fc1 = MyLinear(256, 128, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                            bn_momentum_decay=opt.bn_momentum_decay)
        self.fc2 = MyLinear(128, 64, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                            bn_momentum_decay=opt.bn_momentum_decay)
        self.fc3 = MyLinear(64, 1, activation=None, normalization=None)

        self.dropout1 = nn.Dropout(p=self.opt.dropout)
        self.dropout2 = nn.Dropout(p=self.opt.dropout)

    def forward(self, x, sn=None, epoch=None):
        '''

        :param x: BxMx3 som nodes / Bx3xN points
        :param sn: Bx3xN surface normal
        :return:
        '''

        first_pn_out = self.first_pointnet(x, epoch)  # BxCxN
        feature_1, _ = torch.max(first_pn_out, dim=2, keepdim=False)  # BxC


        second_pn_out = self.second_pointnet(torch.cat((first_pn_out, feature_1.unsqueeze(2).expand_as(first_pn_out)), dim=1), epoch)
        feature_2, _ = torch.max(second_pn_out, dim=2, keepdim=False)

        # get sin(theta)
        fc1_out = self.fc1(feature_2, epoch)
        if self.opt.dropout > 0.1:
            fc1_out = self.dropout1(fc1_out)
        self.fc2_out = self.fc2(fc1_out, epoch)
        if self.opt.dropout > 0.1:
            self.fc2_out = self.dropout2(self.fc2_out)

        sin_theta = torch.tanh(self.fc3(self.fc2_out, epoch))  # Bx1

        return sin_theta


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        # transformer
        self.transformer = Transformer(opt)

        # first PointNet
        if self.opt.surface_normal == True:
            self.first_pointnet = PointResNet(6, [64, 128, 256, 384], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            self.first_pointnet = PointResNet(3, [64, 128, 256, 384], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        if self.opt.som_k >= 2:
            # second PointNet
            self.knnlayer = KNNModule(3+384, (512, 512), activation=self.opt.activation, normalization=self.opt.normalization,
                                      momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

            # final PointNet
            self.final_pointnet = PointNet(3+512, (768, self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization,
                                           momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            # final PointNet
            self.final_pointnet = PointResNet(3+384, (512, 512, 768, self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)



        # build som for clustering, node initalization is done in __init__
        rows = int(math.sqrt(self.opt.node_num))
        cols = rows
        self.som_builder = som.BatchSOM(rows, cols, 3, self.opt.gpu_ids[0], self.opt.batch_size)

        # masked max
        # self.masked_max = operations.MaskedMax(self.opt.node_num)

        # padding
        self.zero_pad = torch.nn.ZeroPad2d(padding=1)

    def forward(self, x, sn, node, node_knn_I, is_train=False, epoch=None):
        '''

        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param node_knn_I: BxMxk_som LongTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''

        # optimize the som, access the Tensor's tensor, the optimize function should not modify the tensor
        # self.som_builder.optimize(x.data)
        # self.som_builder.node.resize_(node.size()).copy_(node)

        # modify the x according to the nodes, minus the center
        mask, mask_row_max, min_idx = som.query_topk(node, x.data, node.size()[2], k=self.opt.k)  # BxkNxnode_num, Bxnode_num, BxkN
        mask_row_sum = torch.sum(mask, dim=1)  # Bxnode_num
        mask = mask.unsqueeze(1)  # Bx1xkNxnode_num

        # if necessary, stack the x
        x_list, sn_list = [], []
        for i in range(self.opt.k):
            x_list.append(x)
            sn_list.append(sn)
        x_stack = torch.cat(tuple(x_list), dim=2)
        sn_stack = torch.cat(tuple(sn_list), dim=2)

        # re-compute center, instead of using som.node
        x_stack_data_unsqueeze = x_stack.data.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (mask_row_sum.unsqueeze(1).float()+1e-5)  # BxCxnode_num
        som_node_cluster_mean = cluster_mean

        # assign each point with a center
        node_expanded = som_node_cluster_mean.unsqueeze(2)  # BxCx1xnode_num, som.node is BxCxnode_num
        centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN

        x_decentered = (x_stack - centers).detach()  # Bx3xkN
        x_augmented = torch.cat((x_decentered, sn_stack), dim=1)  # Bx6xkN

        # go through the first PointNet
        if self.opt.surface_normal == True:
            first_pn_out = self.first_pointnet(x_augmented, epoch)
        else:
            first_pn_out = self.first_pointnet(x_decentered, epoch)

        # gather_index = self.masked_max.compute(first_pn_out, min_idx, mask).detach()
        M = node.size()[2]
        with torch.cuda.device(first_pn_out.get_device()):
            gather_index = index_max.forward_cuda(first_pn_out.detach(),
                                                  min_idx.int(),
                                                  M).detach().long()
        first_pn_out_masked_max = first_pn_out.gather(dim=2, index=gather_index * mask_row_max.unsqueeze(1).long())  # BxCxM

        if self.opt.som_k >= 2:
            # second pointnet, knn search on SOM nodes: ----------------------------------
            knn_center_1, knn_feature_1 = self.knnlayer(som_node_cluster_mean,
                                                        first_pn_out_masked_max,
                                                        node_knn_I,
                                                        self.opt.som_k,
                                                        self.opt.som_k_type,
                                                        epoch)

            # final pointnet --------------------------------------------------------------
            final_pn_out = self.final_pointnet(torch.cat((knn_center_1, knn_feature_1), dim=1), epoch)  # Bx1024xM
        else:
            # final pointnet --------------------------------------------------------------
            final_pn_out = self.final_pointnet(torch.cat((som_node_cluster_mean, first_pn_out_masked_max), dim=1), epoch)  # Bx1024xM

        feature, _ = torch.max(final_pn_out, dim=2, keepdim=False)

        return feature


class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        # classifier
        self.fc1 = MyLinear(self.feature_num, 512, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.fc2 = MyLinear(512, 256, activation=self.opt.activation, normalization=self.opt.normalization,
                            momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        self.fc3 = MyLinear(256, self.opt.classes, activation=None, normalization=None)

        self.dropout1 = nn.Dropout(p=self.opt.dropout)
        self.dropout2 = nn.Dropout(p=self.opt.dropout)

    def forward(self, feature, epoch=None):
        fc1_out = self.fc1(feature, epoch)
        if self.opt.dropout > 0.1:
            fc1_out = self.dropout1(fc1_out)
        self.fc2_out = self.fc2(fc1_out, epoch)
        if self.opt.dropout > 0.1:
            self.fc2_out = self.dropout2(self.fc2_out)
        score = self.fc3(self.fc2_out, epoch)

        return score


