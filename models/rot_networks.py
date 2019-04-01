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
from . import rotation_groups
from .layers import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import index_max


class RotEncoder(nn.Module):
    def __init__(self, opt):
        super(RotEncoder, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        # first PointNet
        if self.opt.surface_normal == True:
            self.first_pointnet = PointResNet(6, [64, 128, 256, 384], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            self.first_pointnet = PointResNet(3, [64, 128, 256, 384], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        if self.opt.som_k >= 2:
            if self.opt.rot_equivariant_pooling_mode == 'per-hierarchy':
                # second PointNet
                self.knnlayer = KNNModule(3 + 384*2, (512, 512), activation=self.opt.activation,
                                          normalization=self.opt.normalization,
                                          momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                          bn_momentum_decay=opt.bn_momentum_decay)

                # final PointNet
                self.final_pointnet = PointNet(3 + 512*2, (768, self.feature_num), activation=self.opt.activation,
                                               normalization=self.opt.normalization,
                                               momentum=opt.bn_momentum,
                                               bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                               bn_momentum_decay=opt.bn_momentum_decay)
            else:
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

        # === rotation equivariant, configure the rotation matrix === begin ===
        self.rotation_matrix_template = torch.zeros((1, self.opt.rot_equivariant_no, 3, 3), dtype=torch.float32)  # 1xRx3x3
        self.rotation_matrix_template[0, ...].copy_(rotation_groups.get_rotation_group_3x3(self.opt.rot_equivariant_mode, self.opt.rot_equivariant_no))
        # === rotation equivariant, configure the rotation matrix === end ===

        # debug for DataParallel
        # self.pn = PointNet(3, (64, 128, 256, 512, self.feature_num), activation=self.opt.activation,
        #                                normalization=self.opt.normalization,
        #                                momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step,
        #                                bn_momentum_decay=opt.bn_momentum_decay)


    def forward(self, x, sn, node, node_knn_I, is_train=False, epoch=None):
        '''

        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param node_knn_I: BxMxk_som LongTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''
        device = x.device

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
        x_stack = torch.cat(tuple(x_list), dim=2)  # Bx3xkN
        sn_stack = torch.cat(tuple(sn_list), dim=2)  # Bx3xkN

        # re-compute center, instead of using som.node
        x_stack_data_unsqueeze = x_stack.data.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (mask_row_sum.unsqueeze(1).float()+1e-5)  # BxCxnode_num
        som_node_cluster_mean = cluster_mean

        # ====== rotate the pc, sn & som_node into R number of rotated versions ======
        B, R, N, kN, M = x_stack.size()[0], \
                         self.opt.rot_equivariant_no, \
                         x.size()[2], x_stack.size()[2], \
                         node.size()[2]
        rotation_matrix = self.rotation_matrix_template.to(device).expand(B, R, 3, 3).detach()  # 1xRx3x3 -> BxRx3x3

        x_stack_rot = torch.matmul(rotation_matrix, x_stack.unsqueeze(1).expand(B, R, 3, kN))  # BxRx3x3 * BxRx3xkN -> BxRx3xkN
        sn_stack_rot = torch.matmul(rotation_matrix, sn_stack.unsqueeze(1).expand(B, R, 3, kN))  # BxRx3xkN
        som_node_rot = torch.matmul(rotation_matrix, som_node_cluster_mean.unsqueeze(1).expand(B, R, 3, M))  # BxRx3xM

        node_knn_I_rot = node_knn_I.unsqueeze(1).expand(B, R, M, self.opt.som_k).contiguous()  # BxRxMxsom_k
        mask_rot = mask.unsqueeze(1).expand(B, R, 1, kN, M).contiguous()
        min_idx_rot = min_idx.unsqueeze(1).expand(B, R, kN).contiguous()
        mask_row_max_rot = mask_row_max.unsqueeze(1).expand(B, R, M).contiguous()

        # ====== rotate the pc, sn & som_node into R number of rotated versions ======

        # assign each point with a center
        # single rotation ------ begin ------
        # node_expanded = som_node_cluster_mean.unsqueeze(2)  # Bx3x1xM, som.node is Bx3xM
        # centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN
        #
        # x_decentered = (x_stack - centers).detach()  # Bx3xkN
        # x_augmented = torch.cat((x_decentered, sn_stack), dim=1)  # Bx6xkN
        # single rotation ------ end ------

        # multiple rotations ------ begin ------
        node_rot_expanded = som_node_rot.unsqueeze(3)  # BxRx3x1xM, som_node_rot is BxRx3xM
        # mask: Bx1xkNxM -> BxRx1xkNxM, self.centers_rot: BxRx3xkN
        centers_rot = torch.sum(mask_rot.float() * node_rot_expanded, dim=4).detach()  # BxRx3xkN

        x_decentered_rot = (x_stack_rot - centers_rot).detach()  # BxRx3xkN
        x_augmented_rot = torch.cat((x_decentered_rot, sn_stack_rot), dim=2)  # BxRx6xkN
        # multiple rotations ------ end ------

        # go through the first PointNet
        if self.opt.surface_normal == True:
            first_pn_out_rot = self.first_pointnet(
                x_augmented_rot.contiguous().view(B*R, 6, kN).contiguous(),
                epoch)
        else:
            first_pn_out_rot = self.first_pointnet(
                x_decentered_rot.contiguous().view(B*R, 6, kN).contiguous(),
                epoch)
        C = first_pn_out_rot.size()[1]

        with torch.cuda.device(first_pn_out_rot.get_device()):
            gather_index_rot = index_max.forward_cuda(first_pn_out_rot.detach(),
                                                      min_idx_rot.contiguous().view(B * R, kN).contiguous().int(), # BxRxkN-> kNxBxR->kN*BR->BR*kN
                                                      M).detach().long()
        first_pn_out_masked_max_rot = first_pn_out_rot.gather(dim=2,
                                                              index=gather_index_rot * mask_row_max_rot.contiguous().view(B*R, M).contiguous().unsqueeze(1).long())  # BRxCxM

        if self.opt.rot_equivariant_pooling_mode == 'per-hierarchy':
            # first_pn_out_masked_max_rot: BRxCxM
            first_pn_out_masked_max_rot_pool = first_pn_out_masked_max_rot.contiguous().view(B, R, C, M).contiguous()  # BxRxCxM
            first_pn_out_masked_max_rot_pool, _ = torch.max(first_pn_out_masked_max_rot_pool, dim=1, keepdim=True)  # BxRxCxM -> Bx1xCxM
            first_pn_out_masked_max_rot_pool = first_pn_out_masked_max_rot_pool.expand(B, R, C, M).contiguous()  # Bx1xCxM -> BxRxCxM
            first_pn_out_masked_max_rot_pool = first_pn_out_masked_max_rot_pool.contiguous().view(B*R,C,M).contiguous()   # BRxCxM

            first_pn_out_masked_max_rot = torch.cat((first_pn_out_masked_max_rot, first_pn_out_masked_max_rot_pool), dim=1)
        if self.opt.som_k >= 2:
            # second pointnet, knn search on SOM nodes: ----------------------------------
            knn_center_1_rot, knn_feature_1_rot = self.knnlayer(som_node_rot.contiguous().view(B*R, 3, M).contiguous(),
                                                                first_pn_out_masked_max_rot,
                                                                node_knn_I_rot.contiguous().view(B*R, M, self.opt.som_k).contiguous(),
                                                                self.opt.som_k,
                                                                self.opt.som_k_type,
                                                                epoch)
            C2 = knn_feature_1_rot.size()[1]

            # final pointnet --------------------------------------------------------------
            if self.opt.rot_equivariant_pooling_mode == 'per-hierarchy':
                knn_feature_1_rot_pool = knn_feature_1_rot.contiguous().view(B, R, C2, M).contiguous()  # B*RxC2xM -> BxRxC2xM
                knn_feature_1_rot_pool, _ = torch.max(knn_feature_1_rot_pool, dim=1, keepdim=True)  # Bx1xC2xM
                knn_feature_1_rot_pool = knn_feature_1_rot_pool.expand(B, R, C2, M).contiguous()  # Bx1xC2xM -> BxRxC2xM
                knn_feature_1_rot_pool = knn_feature_1_rot_pool.contiguous().view(B*R, C2, M).contiguous()

                knn_feature_1_rot = torch.cat((knn_feature_1_rot, knn_feature_1_rot_pool), dim=1)
            final_pn_out_rot = self.final_pointnet(torch.cat((knn_center_1_rot, knn_feature_1_rot), dim=1), epoch)  # Bx1024xM


        else:
            # final pointnet --------------------------------------------------------------
            final_pn_out_rot = self.final_pointnet(torch.cat((som_node_rot.contiguous().view(B*R, 3, M).contiguous(),
                                                              first_pn_out_masked_max_rot),
                                                             dim=1),
                                                   epoch)  # Bx1024xM

        # final_pn_out_rot:  BRx1024xM
        final_pn_out_rot = final_pn_out_rot.contiguous().view(B, R, self.opt.feature_num, M).contiguous()

        feature_rot, _ = torch.max(final_pn_out_rot, dim=3, keepdim=False)  # BxRxC
        feature, _ = torch.max(feature_rot, dim=1, keepdim=False)
        # feature = torch.mean(feature_rot, dim=1, keepdim=False)

        # # debug using vanilla pointnet
        # pn_out = self.pn(x)  # BxCxN
        # feature, _ = torch.max(pn_out, dim=2, keepdim=False)

        return feature


class RotEncoder2D(nn.Module):
    def __init__(self, opt):
        super(RotEncoder2D, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        # first PointNet
        if self.opt.intensity == True:
            self.first_pointnet = PointResNet(3, [64, 64, 128, 128], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            self.first_pointnet = PointResNet(2, [64, 64, 128, 128], activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        if self.opt.som_k >= 2:
            # second PointNet
            self.knnlayer = KNNModule(2+128, (256, 256), activation=self.opt.activation, normalization=self.opt.normalization,
                                      momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

            # final PointNet
            self.final_pointnet = PointNet(2+256, (512, self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization,
                                           momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)
        else:
            # final PointNet
            self.final_pointnet = PointResNet(2+128, (256, 256, 512, self.feature_num), activation=self.opt.activation, normalization=self.opt.normalization,
                                              momentum=opt.bn_momentum, bn_momentum_decay_step=opt.bn_momentum_decay_step, bn_momentum_decay=opt.bn_momentum_decay)

        # build som for clustering, node initalization is done in __init__
        rows = int(math.sqrt(self.opt.node_num))
        cols = rows
        self.som_builder = som.BatchSOM(rows, cols, 2, self.opt.gpu_ids[0], self.opt.batch_size)

        # masked max
        # self.masked_max = operations.MaskedMax(self.opt.node_num)

        # padding
        self.zero_pad = torch.nn.ZeroPad2d(padding=1)

        # === rotation equivariant, configure the rotation matrix === begin ===
        self.rotation_matrix_template = torch.zeros((1, self.opt.rot_equivariant_no, 2, 2), dtype=torch.float32)  # 1xRx2x2
        self.rotation_matrix_template[0, ...].copy_(rotation_groups.get_rotation_group_2x2(self.opt.rot_equivariant_no))
        # === rotation equivariant, configure the rotation matrix === end ===


    def forward(self, x, intensity, node, node_knn_I, is_train=False, epoch=None):
        '''

        :param x: Bx2xN Tensor
        :param intensity: Bx1xN Tensor
        :param node: Bx2xM FloatTensor
        :param node_knn_I: BxMxk_som LongTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''
        device = x.device

        # optimize the som, access the Tensor's tensor, the optimize function should not modify the tensor
        # self.som_builder.optimize(x.data)
        # self.som_builder.node.resize_(node.size()).copy_(node)

        # modify the x according to the nodes, minus the center
        mask, mask_row_max, min_idx = som.query_topk(node, x.data, node.size()[2], k=self.opt.k)  # BxkNxnode_num, Bxnode_num, BxkN
        mask_row_sum = torch.sum(mask, dim=1)  # Bxnode_num
        mask = mask.unsqueeze(1)  # Bx1xkNxnode_num

        # if necessary, stack the x
        x_list, intensity_list = [], []
        for i in range(self.opt.k):
            x_list.append(x)
            intensity_list.append(intensity)
        x_stack = torch.cat(tuple(x_list), dim=2)  # Bx2xkN
        intensity_stack = torch.cat(tuple(intensity_list), dim=2)  # Bx1xkN

        # re-compute center, instead of using som.node
        x_stack_data_unsqueeze = x_stack.data.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (mask_row_sum.unsqueeze(1).float()+1e-5)  # BxCxnode_num
        som_node_cluster_mean = cluster_mean

        # ====== rotate the pc, sn & som_node into R number of rotated versions ======
        B, R, N, kN, M = x_stack.size()[0], \
                         self.opt.rot_equivariant_no, \
                         x.size()[2], x_stack.size()[2], \
                         node.size()[2]
        rotation_matrix = self.rotation_matrix_template.to(device).expand(B, R, 2, 2).detach()  # 1xRx2x2 -> BxRx2x2

        x_stack_rot = torch.matmul(rotation_matrix, x_stack.unsqueeze(1).expand(B, R, 2, kN))  # BxRx2x2 * BxRx2xkN -> BxRx2xkN
        som_node_rot = torch.matmul(rotation_matrix, som_node_cluster_mean.unsqueeze(1).expand(B, R, 2, M))  # BxRx2xM

        intensity_stack_rot = intensity_stack.unsqueeze(1).expand(B, R, 1, kN)  # BxRx1xkN

        node_knn_I_rot = node_knn_I.unsqueeze(1).expand(B, R, M, self.opt.som_k).contiguous()  # BxRxMxsom_k
        mask_rot = mask.unsqueeze(1).expand(B, R, 1, kN, M).contiguous()
        min_idx_rot = min_idx.unsqueeze(1).expand(B, R, kN).contiguous()
        mask_row_max_rot = mask_row_max.unsqueeze(1).expand(B, R, M).contiguous()

        # ====== rotate the pc, sn & som_node into R number of rotated versions ======

        # assign each point with a center
        # single rotation ------ begin ------
        # node_expanded = som_node_cluster_mean.unsqueeze(2)  # Bx3x1xM, som.node is Bx3xM
        # centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN
        #
        # x_decentered = (x_stack - centers).detach()  # Bx3xkN
        # x_augmented = torch.cat((x_decentered, sn_stack), dim=1)  # Bx6xkN
        # single rotation ------ end ------

        # multiple rotations ------ begin ------
        node_rot_expanded = som_node_rot.unsqueeze(3)  # BxRx2x1xM, som_node_rot is BxRx2xM
        # mask: Bx1xkNxM -> BxRx1xkNxM, self.centers_rot: BxRx2xkN
        centers_rot = torch.sum(mask_rot.float() * node_rot_expanded, dim=4).detach()  # BxRx2xkN

        x_decentered_rot = (x_stack_rot - centers_rot).detach()  # BxRx2xkN
        x_augmented_rot = torch.cat((x_decentered_rot, intensity_stack_rot), dim=2)  # BxRx3xkN
        # multiple rotations ------ end ------

        # go through the first PointNet
        if self.opt.intensity == True:
            first_pn_out_rot = self.first_pointnet(
                x_augmented_rot.permute(2, 3, 0, 1).contiguous().view(3, kN, B * R).permute(2, 0, 1).contiguous(),
                epoch)
        else:
            first_pn_out_rot = self.first_pointnet(
                x_decentered_rot.permute(2, 3, 0, 1).contiguous().view(2, kN, B * R).permute(2, 0, 1).contiguous(),
                epoch)
        C = first_pn_out_rot.size()[1]

        with torch.cuda.device(first_pn_out_rot.get_device()):
            gather_index_rot = index_max.forward_cuda(first_pn_out_rot.detach(),
                                                      min_idx_rot.permute(2, 0, 1).contiguous().view(kN, B * R).permute(1, 0).contiguous().int(),  # BxRxkN-> kNxBxR->kN*BR->BR*kN
                                                      M).detach().long()
        first_pn_out_masked_max_rot = first_pn_out_rot.gather(dim=2,
                                                              index=gather_index_rot * mask_row_max_rot.permute(2,0,1).contiguous().view(M, B*R).permute(1,0).contiguous().unsqueeze(1).long())  # BRxCxM

        if self.opt.rot_equivariant_pooling_mode == 'per-hierarchy':
            # first_pn_out_masked_max_rot: BRxCxM
            first_pn_out_masked_max_rot = first_pn_out_masked_max_rot.permute(1,2,0).contiguous().view(C, M, B, R).permute(2,3,0,1).contiguous()  # BxRxCxM
            first_pn_out_masked_max_rot, _ = torch.max(first_pn_out_masked_max_rot, dim=1, keepdim=True)  # BxRxCxM -> Bx1xCxM
            first_pn_out_masked_max_rot = first_pn_out_masked_max_rot.expand(B, R, C, M).contiguous()  # Bx1xCxM -> BxRxCxM
            first_pn_out_masked_max_rot = first_pn_out_masked_max_rot.permute(2,3,0,1).contiguous().view(C,M,B*R).permute(2,0,1).contiguous()   # BRxCxM
        if self.opt.som_k >= 2:
            # second pointnet, knn search on SOM nodes: ----------------------------------
            knn_center_1_rot, knn_feature_1_rot = self.knnlayer(som_node_rot.permute(2,3,0,1).contiguous().view(2, M, B*R).permute(2,0,1).contiguous(),
                                                                first_pn_out_masked_max_rot,
                                                                node_knn_I_rot.permute(2,3,0,1).contiguous().view(M, self.opt.som_k, B*R).permute(2,0,1).contiguous(),
                                                                self.opt.som_k,
                                                                self.opt.som_k_type,
                                                                epoch)
            C2 = knn_feature_1_rot.size()[1]

            # final pointnet --------------------------------------------------------------
            if self.opt.rot_equivariant_pooling_mode == 'per-hierarchy':
                knn_feature_1_rot = knn_feature_1_rot.permute(1,2,0).contiguous().view(C2, M, B, R).permute(2,3,0,1).contiguous()  # B*RxC2xM -> BxRxC2xM
                knn_feature_1_rot, _ = torch.max(knn_feature_1_rot, dim=1, keepdim=True)  # Bx1xC2xM
                knn_feature_1_rot = knn_feature_1_rot.expand(B, R, C2, M).contiguous()  # Bx1xC2xM -> BxRxC2xM
                knn_feature_1_rot = knn_feature_1_rot.permute(2,3,0,1).contiguous().view(C2, M, B*R).permute(2,0,1).contiguous()
            final_pn_out_rot = self.final_pointnet(torch.cat((knn_center_1_rot, knn_feature_1_rot), dim=1), epoch)  # Bx1024xM
        else:
            # final pointnet --------------------------------------------------------------
            final_pn_out_rot = self.final_pointnet(torch.cat((som_node_rot.permute(2,3,0,1).contiguous().view(2, M, B*R).permute(2,0,1).contiguous(),
                                                              first_pn_out_masked_max_rot),
                                                             dim=1),
                                                   epoch)  # Bx1024xM

        # final_pn_out_rot:  BRx1024xM
        final_pn_out_rot = final_pn_out_rot.permute(1,2,0).contiguous().view(self.opt.feature_num, M, B, R).permute(2,3,0,1).contiguous()

        feature_rot, _ = torch.max(final_pn_out_rot, dim=3, keepdim=False)  # BxRxC
        feature, _ = torch.max(feature_rot, dim=1, keepdim=False)

        # # debug using vanilla pointnet
        # pn_out = self.pn(x)  # BxCxN
        # feature, _ = torch.max(pn_out, dim=2, keepdim=False)

        # get statistic of rotation max index
        if feature_rot.size()[0] == 1:
            _, max_idx = torch.max(feature_rot, dim=1, keepdim=False)  # BxC
            # print(max_idx)
            histogram = np.histogram(max_idx.detach().cpu().numpy(), bins=np.asarray(list(range(0, feature_rot.size()[1]+1)))-0.5)
            print(histogram)

        return feature


class RotEncoderFusion(nn.Module):
    def __init__(self, opt):
        super(RotEncoderFusion, self).__init__()
        self.opt = opt
        self.feature_num = opt.feature_num

        # first PointNet
        self.C1 = 128
        if self.opt.surface_normal == True:
            self.first_pointnet = PointNet(6, [int(self.C1/2), int(self.C1/2), int(self.C1/2)],
                                           activation=self.opt.activation,
                                           normalization=self.opt.normalization,
                                           momentum=opt.bn_momentum,
                                           bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                           bn_momentum_decay=opt.bn_momentum_decay)
        else:
            self.first_pointnet = PointNet(3, [int(self.C1/2), int(self.C1/2), int(self.C1/2)],
                                           activation=self.opt.activation,
                                           normalization=self.opt.normalization,
                                           momentum=opt.bn_momentum,
                                           bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                           bn_momentum_decay=opt.bn_momentum_decay)

        self.second_pointnet = PointNet(self.C1, [self.C1, self.C1],
                                        activation=self.opt.activation,
                                        normalization=self.opt.normalization,
                                        momentum=opt.bn_momentum,
                                        bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                        bn_momentum_decay=opt.bn_momentum_decay)

        self.C2 = 512
        if self.opt.som_k >= 2:
            # second PointNet
            self.knnlayer = KNNFusionModule(3 + self.C1,
                                            (int(self.C2 / 2), int(self.C2 / 2), int(self.C2 / 2)),
                                            (self.C2, self.C2),
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            momentum=opt.bn_momentum,
                                            bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                            bn_momentum_decay=opt.bn_momentum_decay)

            # final PointNet
            self.final_pointnet = PointNetFusion(3+self.C2, (512, 512, 512), (self.feature_num, self.feature_num),
                                                 activation=self.opt.activation,
                                                 normalization=self.opt.normalization,
                                                 momentum=opt.bn_momentum,
                                                 bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                                 bn_momentum_decay=opt.bn_momentum_decay)
        else:
            # final PointNet
            self.final_pointnet = PointNetFusion(3+self.C1, (256, 512, 512), (self.feature_num, self.feature_num),
                                                 activation=self.opt.activation,
                                                 normalization=self.opt.normalization,
                                                 momentum=opt.bn_momentum,
                                                 bn_momentum_decay_step=opt.bn_momentum_decay_step,
                                                 bn_momentum_decay=opt.bn_momentum_decay)

        # build som for clustering, node initalization is done in __init__
        rows = int(math.sqrt(self.opt.node_num))
        cols = rows
        self.som_builder = som.BatchSOM(rows, cols, 3, self.opt.gpu_ids[0], self.opt.batch_size)

        # masked max
        # self.masked_max = operations.MaskedMax(self.opt.node_num)

        # padding
        self.zero_pad = torch.nn.ZeroPad2d(padding=1)

        # === rotation equivariant, configure the rotation matrix === begin ===
        self.rotation_matrix_template = torch.zeros((1, self.opt.rot_equivariant_no, 3, 3), dtype=torch.float32)  # 1xRx3x3
        self.rotation_matrix_template[0, ...].copy_(rotation_groups.get_rotation_group_3x3(self.opt.rot_equivariant_mode, self.opt.rot_equivariant_no))
        # === rotation equivariant, configure the rotation matrix === end ===

    def forward(self, x, sn, node, node_knn_I, is_train=False, epoch=None):
        '''

        :param x: Bx3xN Tensor
        :param sn: Bx3xN Tensor
        :param node: Bx3xM FloatTensor
        :param node_knn_I: BxMxk_som LongTensor
        :param is_train: determine whether to add noise in KNNModule
        :return:
        '''
        device = x.device

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
        x_stack = torch.cat(tuple(x_list), dim=2)  # Bx3xkN
        sn_stack = torch.cat(tuple(sn_list), dim=2)  # Bx3xkN

        # re-compute center, instead of using som.node
        x_stack_data_unsqueeze = x_stack.data.unsqueeze(3)  # BxCxkNx1
        x_stack_data_masked = x_stack_data_unsqueeze * mask.float()  # BxCxkNxnode_num
        cluster_mean = torch.sum(x_stack_data_masked, dim=2) / (mask_row_sum.unsqueeze(1).float()+1e-5)  # BxCxnode_num
        som_node_cluster_mean = cluster_mean

        # ====== rotate the pc, sn & som_node into R number of rotated versions ======
        B, R, N, kN, M = x_stack.size()[0], \
                         self.opt.rot_equivariant_no, \
                         x.size()[2], x_stack.size()[2], \
                         node.size()[2]
        rotation_matrix = self.rotation_matrix_template.to(device).expand(B, R, 3, 3).detach()  # 1xRx3x3 -> BxRx3x3

        x_stack_rot = torch.matmul(rotation_matrix, x_stack.unsqueeze(1).expand(B, R, 3, kN))  # BxRx3x3 * BxRx3xkN -> BxRx3xkN
        sn_stack_rot = torch.matmul(rotation_matrix, sn_stack.unsqueeze(1).expand(B, R, 3, kN))  # BxRx3xkN
        som_node_rot = torch.matmul(rotation_matrix, som_node_cluster_mean.unsqueeze(1).expand(B, R, 3, M))  # BxRx3xM

        node_knn_I_rot = node_knn_I.unsqueeze(1).expand(B, R, M, self.opt.som_k).contiguous()  # BxRxMxsom_k
        mask_rot = mask.unsqueeze(1).expand(B, R, 1, kN, M).contiguous()
        min_idx_rot = min_idx.unsqueeze(1).expand(B, R, kN).contiguous()
        mask_row_max_rot = mask_row_max.unsqueeze(1).expand(B, R, M).contiguous()

        # ====== rotate the pc, sn & som_node into R number of rotated versions ======

        # assign each point with a center
        # single rotation ------ begin ------
        # node_expanded = som_node_cluster_mean.unsqueeze(2)  # Bx3x1xM, som.node is Bx3xM
        # centers = torch.sum(mask.float() * node_expanded, dim=3).detach()  # BxCxkN
        #
        # x_decentered = (x_stack - centers).detach()  # Bx3xkN
        # x_augmented = torch.cat((x_decentered, sn_stack), dim=1)  # Bx6xkN
        # single rotation ------ end ------

        # multiple rotations ------ begin ------
        node_rot_expanded = som_node_rot.unsqueeze(3)  # BxRx3x1xM, som_node_rot is BxRx3xM
        # mask: Bx1xkNxM -> BxRx1xkNxM, self.centers_rot: BxRx3xkN
        centers_rot = torch.sum(mask_rot.float() * node_rot_expanded, dim=4).detach()  # BxRx3xkN

        x_decentered_rot = (x_stack_rot - centers_rot).detach()  # BxRx3xkN
        x_augmented_rot = torch.cat((x_decentered_rot, sn_stack_rot), dim=2)  # BxRx6xkN
        # multiple rotations ------ end ------

        # go through the first PointNet
        if self.opt.surface_normal == True:
            first_pn_out_rot = self.first_pointnet(
                x_augmented_rot.contiguous().view(B*R, 6, kN).contiguous(),
                epoch)
        else:
            first_pn_out_rot = self.first_pointnet(
                x_decentered_rot.contiguous().view(B*R, 6, kN).contiguous(),
                epoch)
        C = first_pn_out_rot.size()[1]

        # permute and reshape the min_idx, mask_rot, mask_row_max_rot
        min_idx_rot = min_idx_rot.contiguous().view(B*R, kN).contiguous()  # BxRxkN-> kNxBxR->kN*BR->BR*kN
        mask_rot = mask_rot.contiguous().view(B*R, 1, kN, M).contiguous()  # BxRx1xkNxM -> 1xkNxMxBxR -> 1xkNxMxBR -> BRx1xkNxM
        mask_row_max_rot = mask_row_max_rot.contiguous().view(B*R, M).contiguous().unsqueeze(1).long()

        # first_gather_index_rot = self.masked_max.compute(first_pn_out_rot,
        #                                                  min_idx_rot,
        #                                                  mask_rot).detach()
        with torch.cuda.device(first_pn_out_rot.get_device()):
            first_gather_index_rot = index_max.forward_cuda(first_pn_out_rot.detach(), min_idx_rot.int(), M).detach().long()
        first_pn_out_masked_max_rot = first_pn_out_rot.gather(dim=2,
                                                              index=first_gather_index_rot * mask_row_max_rot)  # BRxCxM

        # scatter the masked_max back to the kN points
        scattered_first_masked_max = torch.gather(first_pn_out_masked_max_rot,
                                                  dim=2,
                                                  index=min_idx_rot.unsqueeze(1).expand(B*R, first_pn_out_rot.size()[1], kN))  # BRxCxkN
        first_pn_out_fusion = torch.cat((first_pn_out_rot, scattered_first_masked_max), dim=1)  # BRx2CxkN
        second_pn_out = self.second_pointnet(first_pn_out_fusion, epoch)

        # second_gather_index_rot = self.masked_max.compute(second_pn_out,
        #                                                   min_idx_rot,
        #                                                   mask_rot).detach()  # BRxCxM
        with torch.cuda.device(second_pn_out.get_device()):
            second_gather_index_rot = index_max.forward_cuda(second_pn_out.detach(), min_idx_rot.int(), M).detach().long()
        second_pn_out_masked_max_rot = second_pn_out.gather(dim=2,
                                                            index=second_gather_index_rot * mask_row_max_rot)  # BxCxM


        if self.opt.rot_equivariant_pooling_mode == 'per-hierarchy':
            # second_pn_out_masked_max_rot: BRxCxM
            second_pn_out_masked_max_rot = second_pn_out_masked_max_rot.contiguous().view(B, R, C, M).contiguous()  # BxRxCxM
            second_pn_out_masked_max_rot, _ = torch.max(second_pn_out_masked_max_rot, dim=1, keepdim=True)  # BxRxCxM -> Bx1xCxM
            second_pn_out_masked_max_rot = second_pn_out_masked_max_rot.expand(B, R, C, M).contiguous()  # Bx1xCxM -> BxRxCxM
            second_pn_out_masked_max_rot = second_pn_out_masked_max_rot.contiguous().view(B*R,C,M,).contiguous()   # BRxCxM
        if self.opt.som_k >= 2:
            # second pointnet, knn search on SOM nodes: ----------------------------------
            knn_center_1_rot, knn_feature_1_rot = self.knnlayer(som_node_rot.contiguous().view(B*R, 3, M).contiguous(),
                                                                second_pn_out_masked_max_rot,
                                                                node_knn_I_rot.contiguous().view(B*R, M, self.opt.som_k).contiguous(),
                                                                self.opt.som_k,
                                                                self.opt.som_k_type,
                                                                epoch)
            C2 = knn_feature_1_rot.size()[1]

            # final pointnet --------------------------------------------------------------
            if self.opt.rot_equivariant_pooling_mode == 'per-hierarchy':
                knn_feature_1_rot = knn_feature_1_rot.contiguous().view(B, R, C2, M).contiguous()  # B*RxC2xM -> BxRxC2xM
                knn_feature_1_rot, _ = torch.max(knn_feature_1_rot, dim=1, keepdim=True)  # Bx1xC2xM
                knn_feature_1_rot = knn_feature_1_rot.expand(B, R, C2, M).contiguous()  # Bx1xC2xM -> BxRxC2xM
                knn_feature_1_rot = knn_feature_1_rot.contiguous().view(B*R, C2, M).contiguous()
            final_pn_out_rot = self.final_pointnet(torch.cat((knn_center_1_rot, knn_feature_1_rot), dim=1), epoch)  # Bx1024xM
        else:
            # final pointnet --------------------------------------------------------------
            final_pn_out_rot = self.final_pointnet(torch.cat((som_node_rot.contiguous().view(B*R, 3, M).contiguous(),
                                                              second_pn_out_masked_max_rot),
                                                             dim=1),
                                                   epoch)  # Bx1024xM

        # final_pn_out_rot:  BRx1024xM
        final_pn_out_rot = final_pn_out_rot.contiguous().view(B, R, self.opt.feature_num, M).contiguous()

        feature_rot, _ = torch.max(final_pn_out_rot, dim=3, keepdim=False)  # BxRxC
        feature, _ = torch.max(feature_rot, dim=1, keepdim=False)

        # # debug using vanilla pointnet
        # pn_out = self.pn(x)  # BxCxN
        # feature, _ = torch.max(pn_out, dim=2, keepdim=False)

        return feature