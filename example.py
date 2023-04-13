import torch
import numpy as np

counter = 100
filtered_flow_coordinates = np.load('mask_index/filtered_flow_coordinates{}.npy'.format(counter))
filtered_flow_coordinates = np.floor_divide(filtered_flow_coordinates, 4)

w = 160
h = 120
patches_per_image = 96
n = 1

x = torch.empty([n, patches_per_image], device="cuda")
y = torch.empty([n, patches_per_image], device="cuda")


for n_ind in range(n):
    for patch_ind in range(patches_per_image):
        x1 = torch.randint(1, w-1, size=[1, 1], device="cuda")
        y1 = torch.randint(1, h-1, size=[1, 1], device="cuda")

        # print(x1 in filtered_flow_coordinates[:,0])
        if x1 in filtered_flow_coordinates[:,0]:
            idx = (filtered_flow_coordinates[:,0] == x).nonzero().flatten()
            
            while not y1 in filtered_flow_coordinates[idx, 1]:
                y1 = torch.randint(1, h-1, size=[1, 1], device="cuda")

        # y_duplicate = torch.tensor(x_ind.size())
        # for matching_ind in x_ind:
        #     while y[matching_ind] in y_duplicate:
        #         y1 = torch.randint(1, h-1, size=[1, 1], device="cuda")
        

        
        # while torch.isin(x1, filtered_flow_coordinates[:,0]): 
        #     x1 = torch.randint(1, w-1, size=[1, 1], device="cuda")
        # while torch.isin(y1, filtered_flow_coordinates[:,1]):
        #     y1 = torch.randint(1, h-1, size=[1, 1], device="cuda")
        # while not x1 in x: 
        #     x1 = torch.randint(1, w-1, size=[1, 1], device="cuda")
        # while not y1 in y:
        #     y1 = torch.randint(1, h-1, size=[1, 1], device="cuda")

        x[n_ind, patch_ind] = x1
        y[n_ind, patch_ind] = y1

# t = torch.Tensor([1, 1, 2, 3])
# ind1 = (t == 1).nonzero(as_tuple=True)[0]
# y_duplicate = torch.empty(ind1.numel())
# print(y_duplicate)
# print(np.shape(filtered_flow_coordinates))