import numpy as np
import glob
import nibabel as nib 
import scipy.io as sio
import os
import imageio
from WL import *
import cv2 
import PIL.Image as Image
from WL import *

def make_dataset(root_liver,root_mask):
    imgs=[]
    #n=len(os.listdir(root_liver))//2
    n = len(os.listdir(root_liver))
    for i in range(n):
        if i==0:
            continue
        img=os.path.join(root_liver,"%04d.dcm"%i)
        mask=os.path.join(root_mask,"liver (%d).bmp"%i)
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

        #img_x直接去读取dicom文件，并且对窗位、窗宽进行调整
        img_x = pydicom.dcmread(x_path)
        img_x = WL(img_x,150,300)
        img_x = Image.fromarray(img_x)
        img_x = img_x.convert('L')
        
        
        img_y = Image.open(y_path)
        img_y = img_y.convert('L')#位深度24转换为8，转为灰度图，与网络的输入输出有关
        
        
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)






