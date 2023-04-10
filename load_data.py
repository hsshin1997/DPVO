import cv2
import glob
import os
import datetime
import numpy as np
import os.path as osp
from pathlib import Path

from copy import deepcopy

import numpy as np
import torch
import matplotlib.pyplot as plt

import evo
from evo.core import sync
import evo.main_ape as main_ape
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import PoseRelation

test_split = \
    ["MH%03d"%i for i in range(8)] + \
    ["ME%03d"%i for i in range(8)]

STRIDE = 1
fx, fy, cx, cy = [320, 320, 320, 240]

def make_traj(args) -> PoseTrajectory3D:
    if isinstance(args, tuple):
        traj, tstamps = args
        return PoseTrajectory3D(positions_xyz=traj[:,:3], orientations_quat_wxyz=traj[:,3:], timestamps=tstamps)
    assert isinstance(args, PoseTrajectory3D), type(args)
    return deepcopy(args)


if __name__ == '__main__':

    # load ground truth (reference) trajectory
    traj_ref = 'P006/pose_left.txt'
    traj_ref = np.loadtxt(traj_ref, delimiter=" ")[::STRIDE,[1, 2, 0, 4, 5, 3, 6]]

    # load DPVO (estimate) trajectory 
    poses= np.loadtxt('poses.txt', dtype=np.float)
    tstamps = np.loadtxt('tstamps.txt', dtype=np.float)

    # poses = torch.load("./poses.pt")
    # poses= np.loadtxt('poses.txt', dtype=float)
    # timestamps = torch.load("./tstamps.pt")
    # original_DPVO = torch.load("./original.pt")


    # filename = 'P006/pose_left.txt'
    # ground_truth_data = np.loadtxt(filename, delimiter=' ', skiprows=1, dtype=float)

    # traj_est = PoseTrajectory3D(
    #         positions_xyz=poses[:,:3],
    #         orientations_quat_wxyz=poses[:,3:],
    #         timestamps=timestamps.cpu())

    # traj_ref = PoseTrajectory3D(
    #         positions_xyz=ground_truth_data[:,:3],
    #         orientations_quat_wxyz=ground_truth_data[:,3:],
    #         timestamps=timestamps.cpu())

    # traj_est = make_traj(traj_est)
    # traj_ref = make_traj(traj_ref)


    # traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    # print(traj_est.positions_xyz)
    # print(data.size())
    # print(original_DPVO.size())

    # def ate(traj_ref, traj_est, timestamps):
        

    #     

    #     traj_ref = PoseTrajectory3D(
    #         positions_xyz=traj_ref[:,:3],
    #         orientations_quat_wxyz=traj_ref[:,3:],
    #         timestamps=timestamps)
        
    #     result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
    #         pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    #     return result.stats["rmse"]

    # print(traj_est.positions_xyz)
    # plt.plot(traj_ref.positions_xyz[:,0], traj_ref.positions_xyz[:,1], label = "Ground Truth")
    # # plt.plot(original_DPVO[:,2].cpu(), original_DPVO[:,0].cpu(), label = "DPVO")
    # # plt.plot(poses[:,0].cpu(), poses[:,1].cpu(), label = "Patch Improvement")
    # plt.plot(traj_est.positions_xyz[:,0], traj_est.positions_xyz[:,1], label = "Patch Improvement")
    # plt.legend()
    # plt.show()