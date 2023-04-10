import numpy as np
import torch

import sys
from PIL import Image, ImageDraw





if __name__ == '__main__':
    patch0 = torch.load('patch_folder/0.pt')
    patch1 = torch.load('patch_folder/1.pt')
    patch2 = torch.load('patch_folder/2.pt')
    patch3 = torch.load('patch_folder/3.pt')
    patch4 = torch.load('patch_folder/4.pt')

    best_patch = torch.load('patch_folder/best_patches.pt')

    print(best_patch[0, 0, :, 1, 1])

    with Image.open("../.jpg") as im:

        draw = ImageDraw.Draw(im)
        draw.line((0, 0) + im.size, fill=128)
        draw.line((0, im.size[1], im.size[0], 0), fill=128)

        # write to stdout
        im.save(sys.stdout, "PNG")