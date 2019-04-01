import numpy as np
import torch
import math
import random
from data import augmentation

#  ============================ R is 3x3 matrix for point clouds ===================================================
def generate_rotation_group(axis_number=3, right_angle_rotation_number=3, negative_reflection=False, include_identity=True):
    '''

    :param axis_number:
    :param right_angle_rotation_number:
    :param negative_reflection: if True, right_angle_array = [right_angle_array, -1*right_angle_array]
    :param include_identity: if False, remove identity from right_angle_array
    :return: numpy array of Nx3x3
    '''
    right_angle_array = np.zeros((axis_number*right_angle_rotation_number+1, 3))
    for axis in range(axis_number):
        for angle in range(right_angle_rotation_number):
            right_angle_array[axis*right_angle_rotation_number+angle, axis] = 90 * (angle+1)
    if False == include_identity:
        right_angle_array = right_angle_array[0:right_angle_array.shape[0]-1, :]
    if negative_reflection:
        right_angle_array = np.append(right_angle_array, -1*right_angle_array, axis=0)
    # print('right_angle_array:')
    # print(right_angle_array)
    # print(right_angle_array.shape)
    # degree to radian
    right_angle_array = right_angle_array * np.pi / 180.0

    rotation_group = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    for i in range(right_angle_array.shape[0]):
        for j in range(right_angle_array.shape[0]):
            rot_i = augmentation.angles2rotation_matrix(right_angle_array[i])
            rot_j = augmentation.angles2rotation_matrix(right_angle_array[j])
            rot_ij = np.dot(rot_i, rot_j)

            # check whether rot_ij exits in rotation_group
            is_exist = False
            for r in range(rotation_group.shape[0]):
                if np.allclose(rot_ij, rotation_group[r]):
                    is_exist = True
                    break
            if False == is_exist:
                # rot_ij is not in the group yet, add it into the group
                rotation_group = np.append(rotation_group, np.expand_dims(rot_ij, axis=0), axis=0)
    # print('rotation_group:')
    # print(rotation_group)
    # print(rotation_group.shape)

    return rotation_group


def verify_rotation_group_3x3(rotation_matrix):
    '''
    check: 1, it forms a group
           2, each 3x3 matrix is orthogonal
    :param rotation_matrix: Rx3x3 tensor
    :return: True / False
    '''
    rotation_matrix_np = rotation_matrix.numpy()
    R = rotation_matrix_np.shape[0]
    I = np.identity(3)
    atol = 1e-6
    # orthogonal
    for r in range(R):
        rrT = np.dot(rotation_matrix_np[r, :, :], np.transpose(rotation_matrix_np[r, :, :]))
        if False == np.allclose(rrT, I, atol=atol):
            return False

    # group
    for i in range(R):
        rot1 = rotation_matrix_np[i]
        for j in range(R):
            rot2 = rotation_matrix_np[j]
            rot_query = np.dot(rot1, rot2)
            # search in the group
            is_in_group = False
            for r in range(R):
                rot_db = rotation_matrix_np[r, :, :]
                if np.allclose(rot_query, rot_db, atol=atol):
                    is_in_group = True
                    break
            if False == is_in_group:
                return False

    return True


def rotation_matrix_2d(R):
    '''
    :param R: rotation number
    :return:
    '''
    rotation_matrix = torch.zeros((R, 3, 3), dtype=torch.float32)
    theta_array = np.asarray(list(range(0, 360, round(360 / R))), dtype=np.float32) * (math.pi / 180.0)  # R-array
    for r in range(R):
        sin_theta = math.sin(theta_array[r])
        cos_theta = math.cos(theta_array[r])

        rot = torch.Tensor(3, 3).zero_().cpu()  # 3x3
        rot[0, 0] = cos_theta
        rot[0, 2] = sin_theta
        rot[1, 1] = 1
        rot[2, 0] = -1 * sin_theta
        rot[2, 2] = cos_theta

        rotation_matrix[r, ...].copy_(rot)
    return rotation_matrix


def rotation_matrix_3d_24():
    rotation_matrix = torch.tensor([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]],

        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
        [[0, 0, -1], [1, 0, 0], [0, -1, 0]],

        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
        [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],

        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
        [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
        [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],

        [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
        [[0, -1, 0], [0, 0, -1], [1, 0, 0]],

        [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    ], dtype=torch.float32)
    return rotation_matrix


def rotation_matrix_3d_12():
    # rotation_matrix = torch.tensor([
    #     [[], [], []],
    #     [[], [], []],
    #     [[], [], []],
    #     [[], [], []],
    #     [[], [], []],
    #     [[], [], []],
    #     [[], [], []],
    #     [[], [], []],
    #     [[], [], []],
    #     [[], [], []],
    #     [[], [], []],
    #     [[], [], []]
    # ], dtype=torch.float32)
    rotation_matrix_np = generate_rotation_group(axis_number=3, right_angle_rotation_number=1, negative_reflection=True, include_identity=False)
    rotation_matrix_np = rotation_matrix_np.astype(np.float32)
    rotation_matrix = torch.from_numpy(rotation_matrix_np)

    return rotation_matrix


def rotation_matrix_3d_4():
    rotation_matrix = torch.tensor([
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    ], dtype=torch.float32)
    return rotation_matrix


def get_rotation_group_3x3(mode, R):
    '''
    rotation group
    :param mode: '2d' / '3d'
    :param R: rotation_number
    :return: a CPU tensor of Rx3x3
    '''
    rotation_matrix = None
    if '2d' == mode:
        rotation_matrix = rotation_matrix_2d(R)
    elif '3d' == mode:
        if 24 == R:
            rotation_matrix =  rotation_matrix_3d_24()
        elif 12 == R:
            rotation_matrix = rotation_matrix_3d_12()
        elif 4 == R:
            rotation_matrix = rotation_matrix_3d_4()
        else:
            raise Exception('Please input a valid rotation group.')
    else:
        raise Exception('Please input a valid rotation group.')

    # verify that the rotation_matrix is a valid rotation group
    if False == verify_rotation_group_3x3(rotation_matrix):
        raise Exception("The rotation matrices do not form a valid rotation group.")

    # debug
    # print(rotation_matrix)
    # print(rotation_matrix.size())
    return rotation_matrix



#  ============================ R is 2x2 matrix for images ===================================================
def verify_rotation_group_2x2(rotation_matrix):
    '''
    check: 1, it forms a group
           2, each 2x2 matrix is orthogonal
    :param rotation_matrix: Rx2x2 tensor
    :return: True / False
    '''
    rotation_matrix_np = rotation_matrix.numpy()
    R = rotation_matrix_np.shape[0]
    I = np.identity(2)
    atol = 1e-6
    # orthogonal
    for r in range(R):
        rrT = np.dot(rotation_matrix_np[r, :, :], np.transpose(rotation_matrix_np[r, :, :]))
        if False == np.allclose(rrT, I, atol=atol):
            return False

    # group
    for i in range(R):
        rot1 = rotation_matrix_np[i]
        for j in range(R):
            rot2 = rotation_matrix_np[j]
            rot_query = np.dot(rot1, rot2)
            # search in the group
            is_in_group = False
            for r in range(R):
                rot_db = rotation_matrix_np[r, :, :]
                if np.allclose(rot_query, rot_db, atol=atol):
                    is_in_group = True
                    break
            if False == is_in_group:
                return False

    return True


def rotation_matrix_2x2(R):
    '''
    :param R: rotation number
    :return:
    '''
    rotation_matrix = torch.zeros((R, 2, 2), dtype=torch.float32)
    theta_array = np.asarray(list(range(0, 360, round(360 / R))), dtype=np.float32) * (math.pi / 180.0)  # R-array
    for r in range(R):
        sin_theta = math.sin(theta_array[r])
        cos_theta = math.cos(theta_array[r])

        rot = torch.Tensor(2, 2).zero_().cpu()  # 2x2
        rot[0, 0] = cos_theta
        rot[0, 1] = -1 * sin_theta
        rot[1, 0] = sin_theta
        rot[1, 1] = cos_theta

        rotation_matrix[r, ...].copy_(rot)
    return rotation_matrix


def get_rotation_group_2x2(R):
    '''
    rotation group
    :param R: rotation_number
    :return: a CPU tensor of Rx2x2
    '''
    rotation_matrix = rotation_matrix_2x2(R)
    # verify that the rotation_matrix is a valid rotation group
    if False == verify_rotation_group_2x2(rotation_matrix):
        raise Exception("The rotation matrices do not form a valid rotation group.")
    return rotation_matrix



if __name__ == '__main__':
    # # tetrahedral group
    # generate_rotation_group(axis_number=3, right_angle_rotation_number=1, negative_reflection=True, include_identity=False)
    # # cube group
    # generate_rotation_group(axis_number=3, right_angle_rotation_number=3)

    # test group
    rotation_matrix = get_rotation_group('3d', 4)
    is_valid = verify_rotation_group(rotation_matrix)
    print(rotation_matrix)
    print(is_valid)
