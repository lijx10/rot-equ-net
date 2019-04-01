import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import matplotlib.pyplot as plt

# ==================================== 2d function =========================================
def atomic_rotate(data, angle):
    '''

    :param data: numpy array of Nx2 array
    :param angle: float
    :return: rotated_data: numpy array of Nx2
    '''
    R = angles2rotation_matrix(angle)
    rotated_data = np.dot(data, R)

    return rotated_data


def angles2rotation_matrix(angle):
    R = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])
    return R


def rotate_pc_with_som_pytorch_batch(pc, som, rotation_angle=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Bx2xN CPU tensor, original point clouds
        Return:
          Bx2xN CPU tensor, rotated point clouds
    """
    if rotation_angle is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix_np = np.array([[cosval, -sinval],
                                   [sinval, cosval]])
    rotation_matrix = torch.from_numpy(rotation_matrix_np.astype(np.float32)).unsqueeze(0)  # 1x2x2

    rotated_pc = torch.matmul(rotation_matrix, pc)  # 1x2x2 * Bx2xN -> Bx2xN
    rotated_som = torch.matmul(rotation_matrix, som)  # 1x2x2 * Bx2xM -> Bx2xM

    return rotated_pc, rotated_som


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx2 array, original point clouds
        Return:
          Nx2 array, jittered point clouds
    """
    N, C = data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data


def random_rotate_pc_with_som_np(pc, som):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx2 array, original point clouds
        Return:
          Nx2 array, rotated point clouds
    """
    # uniform sampling
    angle = np.random.uniform() * 2 * np.pi

    rotated_pc = atomic_rotate(pc, angle)
    rotated_som = atomic_rotate(som, angle)

    return rotated_pc, rotated_som
