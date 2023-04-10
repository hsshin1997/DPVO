import matplotlib.pyplot as plt
import numpy as np


STRIDE = 1

x_dpvo = np.loadtxt('ransac_results/x_dpvo.txt', dtype=np.float64)
y_dpvo = np.loadtxt('ransac_results/y_dpvo.txt', dtype=np.float64)

x_gt = np.loadtxt('ransac_results/x_gt.txt', dtype=np.float64)
y_gt = np.loadtxt('ransac_results/y_gt.txt', dtype=np.float64)

traj_ref = 'P006/pose_left.txt'
traj_ref = np.loadtxt(traj_ref, delimiter=" ")

plt.plot(traj_ref[:,0], traj_ref[:,1], label = "Ground Truth")
plt.plot(x_dpvo, y_dpvo, label = "Patch Improvement")
plt.legend()
plt.show()