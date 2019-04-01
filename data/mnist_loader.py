import torch.utils.data as data

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

from .augmentation2d import *


# Read numpy array data and label from h5_filename
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def make_dataset_mnist(root, mode, opt):
    # already normalized
    if mode == 'train':
        dataset = np.load(os.path.join(root, 'train.npz'))  # ['som', 'label', 'pc']
        pc_np = dataset['pc'][0:10000]  # Nx512x3
        som_np = dataset['som'][0:10000]  # Nx16x2
        label_np = dataset['label'][0:10000]  # N
    elif mode == 'validation':
        dataset = np.load(os.path.join(root, 'train.npz'))  # ['som', 'label', 'pc']
        pc_np = dataset['pc'][10000:12000]
        som_np = dataset['som'][10000:12000]
        label_np = dataset['label'][10000:12000]
    elif mode == 'test':
        dataset = np.load(os.path.join(root, 'test.npz'))  # ['som', 'label', 'pc']
        pc_np = dataset['pc']
        som_np = dataset['som']
        label_np = dataset['label']
    else:
        raise Exception('Invalid mode.')

    return {'pc': pc_np, 'som': som_np, 'label': label_np}


class KNNBuilder:
    def __init__(self, k):
        self.k = k
        self.dimension = 2

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx2
        :return: Faiss index, in CPU
        '''
        index = faiss.IndexFlatL2(self.dimension)  # dimension is 2
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx2
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        D, I = index.search(query, k)
        return D, I

    def self_build_search(self, x):
        '''

        :param x: numpy array of Nxd
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        x = np.ascontiguousarray(x, dtype=np.float32)
        index = self.build_nn_index(x)
        D, I = self.search_nn(index, x, self.k)
        return D, I


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


class MNIST_Loader(data.Dataset):
    def __init__(self, root, mode, opt):
        super(MNIST_Loader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode

        self.dataset = make_dataset_mnist(self.root, mode, opt)

        # kNN search on SOM nodes
        self.knn_builder = KNNBuilder(self.opt.som_k)

        # farthest point sample
        self.fathest_sampler = FarthestSampler()

    def __len__(self):
        return self.dataset['pc'].shape[0]

    def __getitem__(self, index):
        pc_np = self.dataset['pc'][index, :, 0:2]  # Nx2
        intensity_np = self.dataset['pc'][index, :, 2:]  # Nx1
        som_node_np = self.dataset['som'][index]  # Nx2
        label = self.dataset['label'][index]  # scalar

        # augmentation
        if self.mode == 'train':
            # random 2d rotation
            if self.opt.rot_augmentation:
                pc_np, som_node_np = random_rotate_pc_with_som_np(pc_np, som_node_np)

            # random jittering
            pc_np = jitter_point_cloud(pc_np)
            intensity_np = jitter_point_cloud(intensity_np)
            som_node_np = jitter_point_cloud(som_node_np, sigma=0.04, clip=0.1)

            # random scale
            scale = np.random.uniform(low=0.8, high=1.2)
            pc_np = pc_np * scale
            som_node_np = som_node_np * scale
            intensity_np = intensity_np * scale

            # random shift
            if self.opt.translation_perturbation:
                shift = np.random.uniform(-0.1, 0.1, (1, 2))
                pc_np += shift
                som_node_np += shift

        # convert to tensor
        pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 2xN

        # intensity
        intensity = torch.from_numpy(intensity_np.transpose().astype(np.float32))  # 1xN

        # som
        som_node = torch.from_numpy(som_node_np.transpose().astype(np.float32))  # 2xM

        # kNN search: som -> som
        if self.opt.som_k >= 2:
            D, I = self.knn_builder.self_build_search(som_node_np)
            som_knn_I = torch.from_numpy(I.astype(np.int64))  # M x som_k
        else:
            som_knn_I = torch.from_numpy(np.arange(start=0, stop=self.opt.node_num, dtype=np.int64).reshape((self.opt.node_num, 1)))  # M x 1

        # print(som_node_np)
        # print(D)
        # print(I)
        # assert False

        return pc, intensity, label, som_node, som_knn_I


if __name__=="__main__":
    # dataset = make_dataset_modelnet40('/ssd/dataset/modelnet40_ply_hdf5_2048/', True)
    # print(len(dataset))
    # print(dataset[0])


    class VirtualOpt():
        def __init__(self):
            self.load_all_data = False
            self.input_pc_num = 5000
            self.batch_size = 8
            self.dataset = '10k'
            self.node_num = 64
            self.classes = 10
            self.som_k = 9
    opt = VirtualOpt()
    trainset = ModelNet_Shrec_Loader('/ssd/dataset/modelnet40-normal_numpy/', 'train', opt)
    print('---')
    print(len(trainset))
    print(trainset[0])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
