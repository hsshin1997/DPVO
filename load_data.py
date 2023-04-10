import numpy as np
import torch
import matplotlib.pyplot as plt

poses = torch.load("./poses.pt")
original_DPVO = torch.load("./original.pt")


filename = 'P006/pose_left.txt'
data = np.loadtxt(filename, delimiter=' ', skiprows=1, dtype=float)

# print(data.size())
print(original_DPVO.size())

print(poses)
plt.plot(data[:,0], data[:,1], label = "Ground Truth")
plt.plot(original_DPVO[:,2].cpu(), original_DPVO[:,0].cpu(), label = "DPVO")
plt.plot(poses[:,2].cpu(), poses[:,0].cpu(), label = "Patch Improvement")
plt.legend()
plt.show()