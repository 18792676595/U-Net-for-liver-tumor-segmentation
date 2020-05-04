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



def make_dataset(root_liver,root_mask):
    imgs=[]
    #n=len(os.listdir(root_liver))//2
    n = len(os.listdir(root_liver))
    ls = os.listdir(root_liver)
    for i in range(n):
        img=os.path.join(root_liver,ls[i])
        mask=os.path.join(root_mask,ls[i])
        imgs.append((img,mask))
    return imgs


class LiverDataset(data.Dataset):
    def __init__(self, root_liver,root_mask, transform=None, target_transform=None):
        imgs = make_dataset(root_liver,root_mask)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_x = img_x.resize((512,512))
        img_x = img_x.convert('RGB')
        img_y = Image.open(y_path)
        img_y = img_y.resize((512,512))
        img_y = img_y.convert('RGB')
        
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

