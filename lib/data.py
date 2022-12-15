#!/usr/bin/env python3

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data
import pdb
import imageio
import time
import random
import imageio

from lib.utils import *


class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample
    ):
        self.subsample = subsample
        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)


    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        try:
            return unpack_sdf_samples(filename, self.subsample), idx, self.npyfiles[idx]
        except:
            print ('skipping...', filename)
            return self.__getitem__((idx+1)%self.__len__())


class RGBA2SDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        mesh_name = self.npyfiles[idx].split(".npz")[0]

        # fetch sdf samples
        sdf_filename = os.path.join(
            self.data_source, self.npyfiles[idx]
        )
        sdf_samples = unpack_sdf_samples(sdf_filename,  self.subsample)

        if self.is_train:
            # reset seed for random sampling training data (see https://github.com/pytorch/pytorch/issues/5059)
            np.random.seed( int(time.time()) + idx)
            id = np.random.randint(0, self.num_views)
        else:
            np.random.seed(idx)
            id = np.random.randint(0, self.num_views)

        view_id = '{0:02d}'.format(id)

        image_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), view_id + ".png")
        RGBA = unpack_images(image_filename)

        # fetch cameras
        metadata_filename = os.path.join(self.data_source, mesh_name.replace("samples", "renders"), "rendering_metadata.txt")
        intrinsic, extrinsic = get_camera_matrices(metadata_filename, id)

        return sdf_samples, RGBA, intrinsic, extrinsic, mesh_name


class SketchSDF(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        is_train=False,
        num_views = 1,
    ):
        self.subsample = subsample
        self.is_train = is_train
        self.data_source = data_source
        self.num_views = num_views
        self.npyfiles =  get_instance_filenames(data_source, split)

        # Get point cloud
        pc_4096 = np.load('/scratch/i3d/pc_4096.npy') # 6778 x 4096 x 3
        all_files_names = open('/scratch/i3d/all.txt', 'r').read().split('\n')[:-1] # 6778 x 1
        self.all_point_set = {}
        for idx in range(len(all_files_names)):
            self.all_point_set[all_files_names[idx]] = pc_4096[idx]

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):

        try:
            mesh_name = self.npyfiles[idx].split(".npz")[0]
            mesh_name = os.path.split(mesh_name)[-1]

            # fetch sdf samples
            sdf_filename = os.path.join(
                self.data_source, self.npyfiles[idx]
            )
            sdf_samples = unpack_sdf_samples(sdf_filename,  self.subsample)

            point_set = self.all_point_set[mesh_name] # 4096 x 3
            point_set = torch.from_numpy(point_set).transpose(1, 0) # 3 x 4096

            return sdf_samples, point_set, mesh_name

        except:
            print ('skipping...', mesh_name)
            return self.__getitem__((idx+1)%self.__len__())
