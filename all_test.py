import numpy as np
import glob
import nibabel as nib 
import scipy.io as sio
import os
import imageio
from WL import *
import cv2 
import PIL.Image as Image
import torch.utils.data as data


data = "data/LITS_train"
ls = os.listdir(data)

root1 = os.path.join(data,ls[0],"raw")
root2 = os.path.join(data,ls[0],"liver_target")
lsr1 = os.listdir(root1)
lsr2 = os.listdir(root2)
flag = False
leng1 = len(lsr1)
leng2 = len(lsr2)


for i in range(leng1):
    if lsr1[i]==lsr2[i]:
        flag = True
    else:
        flag = False
        break
        print(i)


print(flag)