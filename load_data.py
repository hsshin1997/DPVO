import numpy as np
import torch
import matplotlib.pyplot as plt

poses = torch.load("./poses.pt")


filename = 'P006/pose_left.txt'
data = np.loadtxt(filename, delimiter=' ', skiprows=1, dtype=float)
# print(data)

plt.plot(data[:,0], data[:,1], label = "ground truth")
plt.plot(poses[:,0].cpu(), poses[:,1].cpu(), label = "method 1")
plt.show()