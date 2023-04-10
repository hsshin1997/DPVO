import numpy as np
import torch
import matplotlib.pyplot as plt

import evo
import evo.main_ape as main_ape
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import PoseRelation

poses = torch.load("./poses.pt")
# original_DPVO = torch.load("./original.pt")


filename = 'P006/pose_left.txt'
data = np.loadtxt(filename, delimiter=' ', skiprows=1, dtype=float)

traj_est = PoseTrajectory3D(
        positions_xyz=data[:,:3],
        orientations_quat_wxyz=data[:,3:],
        timestamps=timestamps)

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

print(poses)
plt.plot(data[:,0], data[:,1], label = "Ground Truth")
# plt.plot(original_DPVO[:,2].cpu(), original_DPVO[:,0].cpu(), label = "DPVO")
plt.plot(poses[:,0].cpu(), poses[:,1].cpu(), label = "Patch Improvement")
plt.legend()
plt.show()