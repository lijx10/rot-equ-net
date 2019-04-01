import random
import numbers
import os
import os.path
import numpy as np
import struct
import math

import torch
import torchvision
import matplotlib.pyplot as plt
import h5py
import faiss

# ============================= 3d function =======================================
def rotate_point_cloud_with_normal_som_pytorch_batch_3d(pc, surface_normal, som, rotation_angles=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Bx3xN CPU tensor, original point clouds
        Return:
          Bx3xN CPU tensor, rotated point clouds
    """
    if rotation_angles is None:
        rotation_angles = np.random.uniform(size=3) * 2 * np.pi
    rotation_matrix_np = angles2rotation_matrix(rotation_angles)
    rotation_matrix = torch.from_numpy(rotation_matrix_np.astype(np.float32)).unsqueeze(0)  # 1x3x3

    rotated_pc = torch.matmul(rotation_matrix, pc)  # 1x3x3 * Bx3xN -> Bx3xN
    rotated_sn = torch.matmul(rotation_matrix, surface_normal)  # 1x3x3 * Bx3xN -> Bx3xN
    rotated_som = torch.matmul(rotation_matrix, som)  # 1x3x3 * Bx3xM -> Bx3xM

    return rotated_pc, rotated_sn, rotated_som


def rotate_point_cloud_with_normal_som_pytorch_batch(pc, surface_normal, som, rotation_angle=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Bx3xN CPU tensor, original point clouds
        Return:
          Bx3xN CPU tensor, rotated point clouds
    """
    if rotation_angle is None:
        rotation_angle = np.random.uniform() * 2 * np.pi
    # rotation_angle = np.random.randint(low=0, high=12) * (2*np.pi / 12.0)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix_np = np.array([[cosval, 0, sinval],
                                   [0, 1, 0],
                                   [-sinval, 0, cosval]])
    rotation_matrix = torch.from_numpy(rotation_matrix_np.astype(np.float32)).unsqueeze(0)  # 1x3x3

    rotated_pc = torch.matmul(rotation_matrix, pc)  # 1x3x3 * Bx3xN -> Bx3xN
    rotated_sn = torch.matmul(rotation_matrix, surface_normal)  # 1x3x3 * Bx3xN -> Bx3xN
    rotated_som = torch.matmul(rotation_matrix, som)  # 1x3x3 * Bx3xM -> Bx3xM

    return rotated_pc, rotated_sn, rotated_som


def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def atomic_rotate(data, angles):
    '''

    :param data: numpy array of Nx3 array
    :param angles: numpy array / list of 3
    :return: rotated_data: numpy array of Nx3
    '''
    R = angles2rotation_matrix(angles)
    rotated_data = np.dot(data, R)

    return rotated_data


def rotate_point_cloud_90(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    y_angle = np.random.randint(low=0, high=4) * (np.pi/2.0)
    angles = [0, y_angle, 0]
    rotated_data = atomic_rotate(data, angles)

    return rotated_data


def rotate_point_cloud_up(data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    y_angle = np.random.uniform() * 2 * np.pi
    angles = [0, y_angle, 0]
    rotated_data = atomic_rotate(data, angles)

    return rotated_data


def rotate_point_cloud_up_with_normal_som(pc, surface_normal, som):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """

    # uniform sampling
    y_angle = np.random.uniform() * 2 * np.pi
    angles = [0, y_angle, 0]

    rotated_pc = atomic_rotate(pc, angles)
    rotated_surface_normal = atomic_rotate(surface_normal, angles)
    rotated_som = atomic_rotate(som, angles)

    return rotated_pc, rotated_surface_normal, rotated_som


def rotate_point_cloud_3d(data):
    # uniform sampling
    angles = np.random.rand(3) * np.pi * 2
    rotated_data = atomic_rotate(data, angles)

    return rotated_data


def rotate_point_cloud_3d_with_normal_som(pc, surface_normal, som):
    # uniform sampling
    angles = np.random.rand(3) * np.pi * 2
    rotated_pc = atomic_rotate(pc, angles)
    rotated_surface_normal = atomic_rotate(surface_normal, angles)
    rotated_som = atomic_rotate(som, angles)

    return rotated_pc, rotated_surface_normal, rotated_som


def rotate_perturbation_point_cloud(data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    # truncated Gaussian sampling
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    rotated_data = atomic_rotate(data, angles)

    return rotated_data


def rotate_perturbation_point_cloud_with_normal_som(pc, surface_normal, som, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    # truncated Gaussian sampling
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    rotated_pc = atomic_rotate(pc, angles)
    rotated_surface_normal = atomic_rotate(surface_normal, angles)
    rotated_som = atomic_rotate(som, angles)

    return rotated_pc, rotated_surface_normal, rotated_som


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, jittered point clouds
    """
    N, C = data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    jittered_data += data
    return jittered_data
