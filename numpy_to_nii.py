import numpy as np
import nibabel as nib 
import scipy.io as sio
import os
import imageio
import cv2 
import PIL.Image as Image
import torch.utils.data as data
import glob



for i in range(12):
    filename = glob.glob("data/challenge/eroded_and_dilate7/%d_*"%(i+1))
    livers  = []
    for name in filename:
        img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
        img[img>=1] = 1
        livers.append(img)
        
    livers = np.asarray(livers)  
    livers = np.transpose(livers, [1,2,0])
    new_image = nib.Nifti1Image(livers, affine=np.eye(4))  
    nib.save(new_image,"D:/BaiduNetdiskDownload/liver/liver_%d/liver_seg.nii"%(i+1))

    
    


