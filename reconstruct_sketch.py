#!/usr/bin/env python3

import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import json
import time
import pdb
import open3d as o3d

import imageio
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturesVertex
)
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes

import lib
import lib.workspace as ws
from lib.utils import *


def main_function(experiment_directory, continue_from,  iterations, marching_cubes_resolution, regularize):

    device=torch.device('cuda:0')
    specs = ws.load_experiment_specifications(experiment_directory)

    print("Reconstruction from experiment description: \n" + ' '.join([str(elem) for elem in specs["Description"]]))

    data_source = specs["DataSource"]
    test_split_file = specs["TestSplit"]

    arch_encoder = __import__("lib.models." + specs["NetworkEncoder"], fromlist=["ResNet"])
    arch_decoder = __import__("lib.models." + specs["NetworkDecoder"], fromlist=["DeepSDF"])
    latent_size = specs["CodeLength"]

    encoder = arch_encoder.PointNetEncoder(k=latent_size).cuda()
    decoder = arch_decoder.DeepSDF(latent_size, **specs["NetworkSpecs"]).cuda()

    encoder = torch.nn.DataParallel(encoder)
    decoder = torch.nn.DataParallel(decoder)

    print("testing with {} GPU(s)".format(torch.cuda.device_count()))

    num_samp_per_scene = specs["SamplesPerScene"]
    with open(test_split_file, "r") as f:
        test_split = json.load(f)

    sdf_dataset_test = lib.data.SketchSDF(
        data_source, test_split, num_samp_per_scene, is_train=True, num_views = specs["NumberOfViews"]
    )
    torch.manual_seed(int( time.time() * 1000.0 ))
    sdf_loader_test = data_utils.DataLoader(
        sdf_dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=False,
    )

    num_scenes = len(sdf_loader_test)
    print("There are {} scenes".format(num_scenes))

    print('Loading epoch "{}"'.format(continue_from))

    print ('Loading model from ', experiment_directory, continue_from)
    ws.load_model_parameters(
        experiment_directory, continue_from, encoder, decoder
    )
    encoder.eval()

    optimization_meshes_dir = os.path.join(
        args.experiment_directory, ws.reconstructions_subdir, str(continue_from)
    )

    if not os.path.isdir(optimization_meshes_dir):
        os.makedirs(optimization_meshes_dir)

    all_cd_dist = []
    for sdf_data, pointset, name, index in sdf_loader_test:

        out_name = name[0]
        # store input stuff
        pointset_filename = os.path.join(optimization_meshes_dir, out_name, "input.ply")
        # skip if it is already there
        if os.path.exists(os.path.dirname(pointset_filename)):
            print(name[0], " exists already ")
            continue
        print('Reconstructing {}...'.format(out_name))

        if not os.path.exists(os.path.dirname(pointset_filename)):
            os.makedirs(os.path.dirname(pointset_filename))

        pointset_export = pointset[0].permute(1, 0).cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointset_export)
        o3d.io.write_point_cloud(pointset_filename, pcd)

        # get latent code from pointset
        latent = encoder(pointset)
        # get estimated mesh
        verts, faces, samples, next_indices = lib.mesh.create_mesh(decoder, latent, N=marching_cubes_resolution, output_mesh = True)

        # store raw output
        mesh_filename = os.path.join(optimization_meshes_dir, out_name, "predicted.ply")
        lib.mesh.write_verts_faces_to_file(verts, faces, mesh_filename)

        verts = torch.from_numpy(verts).type(torch.FloatTensor)
        cd_dist = chamfer_distance(pointset.permute(0, 2, 1), verts[torch.randperm(len(verts))[:4096]].unsqueeze(0))[0]
        print ('CD {}: {}'.format(out_name, cd_dist))
        all_cd_dist.append(cd_dist)

    all_cd_dist = torch.stack(all_cd_dist)
    print ('Chamfer Distance -- mean: {}, std: {}, min: {}, max: {}'.format(
        all_cd_dist.mean(), all_cd_dist.std(), all_cd_dist.min(), all_cd_dist.max()))


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        default="latest",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--resolution",
        default=256,
        help="Marching cubes resolution for reconstructed surfaces.",
    )
    arg_parser.add_argument(
        "--iterations",
        default=100,
        help="Number of refinement iterations.",
    )
    arg_parser.add_argument("--regularize", default=0.0, help="L2 regularization weight on latent vector")

    args = arg_parser.parse_args()
    main_function(args.experiment_directory, args.continue_from, int(args.iterations), int(args.resolution), float(args.regularize))
