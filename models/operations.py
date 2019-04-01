import time
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.multiprocessing as mp
import threading
import ctypes


# generalized batch size
CUDA_SHARED_MEM_DIM_X = 24
# size of SOM
CUDA_SHARED_MEM_DIM_Y = 512





def knn_gather_wrapper(som_node, som_node_knn_I):
    '''

    :param som_node: Bx3xN
    :param som_node_knn_I: BxNxK
    :param som_node_neighbors: Bx3xNxK
    :return:
    '''
    B = som_node.size()[0]
    C = som_node.size()[1]
    N = som_node.size()[2]
    K = som_node_knn_I.size()[2]
    assert C==3 or C==2

    # with numba.cuda.gpus[som_node.device.index]:
    #     som_node_neighbors = torch.zeros((B, C, N, K), dtype=torch.float32, device=som_node.device)
    #
    #     som_node_cuda = get_devicendarray_float32(som_node.data)
    #     som_node_knn_I_cuda = get_devicendarray_int32(som_node_knn_I.int().data)
    #     som_node_neighbors_cuda = get_devicendarray_float32(som_node_neighbors.data)
    #
    #     knn_gather[(N, K), (B, C)](som_node_cuda, som_node_knn_I_cuda, som_node_neighbors_cuda)

    som_node_neighbors = knn_gather_by_indexing(som_node, som_node_knn_I)

    return som_node_neighbors


def knn_gather_by_indexing(som_node, som_node_knn_I):
    '''

    :param som_node: BxCxN
    :param som_node_knn_I: BxNxK
    :param som_node_neighbors: BxCxNxK
    :return:
    '''
    B = som_node.size()[0]
    C = som_node.size()[1]
    N = som_node.size()[2]
    K = som_node_knn_I.size()[2]

    som_node_knn_I = som_node_knn_I.unsqueeze(1).expand(B, C, N, K).contiguous().view(B, C, N*K)
    som_node_neighbors = torch.gather(som_node, dim=2, index=som_node_knn_I).view(B, C, N, K)

    return som_node_neighbors

# ================================== get the k nearest neighbors of the SOM nodes / features ======================================




# ============ test ===========
def get_angles(a, b):
    '''
    calculate the angle between vector a and b
    :param a: Bx3xMxK tensor
    :param b: Bx3xMxK tensor
    :return: Bx1xMxK tensor
    '''
    axb = torch.cross(a, b, dim=1)  # Bx3xMxK
    a_1x3 = a.permute(0, 2, 3, 1).contiguous().unsqueeze(3)  # BxMxKx3 -> BxMxKx1x3
    b_3x1 = b.permute(0, 2, 3, 1).contiguous().unsqueeze(4)  # BxMxKx3 -> BxMxKx3x1
    ab = torch.matmul(a_1x3, b_3x1).squeeze(3).squeeze(3)  # BxMxKx1x1

    angle = torch.atan2(torch.norm(axb, dim=1, keepdim=False), ab).unsqueeze(1)
    return angle


if __name__=='__main__':
    # from kitti.options_detector import Options
    # opt = Options().parse()  # set CUDA_VISIBLE_DEVICES before import torch
    print('Done.')

