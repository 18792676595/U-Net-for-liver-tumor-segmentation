import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from networks.cenet import CE_Net_
import cv2
import nibabel as nib 
import time

x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

y_transforms = transforms.ToTensor()


# 是否使用cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


#liver   'ckp_challenge_only/CE_LITS_and_challenge_model_epoch138.pth'
#tumor   'ckp_challenge_only/CE_LITS_and_challenge_model_epoch142.pth'

def end_to_end(raw_path,liver_model_path,tumor_model_path):

    volume = nib.load(raw_path).get_fdata()
    volume = np.transpose(volume, [2,0,1])
    

    liver_model = CE_Net_(3, 1).to(device)
    liver_model.load_state_dict(torch.load(liver_model_path))
    liver_model.to(device)
    liver_model.eval()

    liver_target = []


    tumor_model = CE_Net_(3, 1).to(device)
    tumor_model.load_state_dict(torch.load(tumor_model_path))
    tumor_model.to(device)
    tumor_model.eval()

    tumor_target = []
    
    for i in range(volume.shape[0]):
        a = volume[i]
        img_x = Image.fromarray(np.int16(np.array(volume[i])))
        h,w = img_x.size
        img_x = img_x.resize((512,512))
        img_x = img_x.convert('RGB')
        img_x = x_transforms(img_x)
        img_x = img_x.to(device)
        img_x = torch.unsqueeze(img_x, 0)

        img_liver = liver_model(img_x)
        img_tumor = tumor_model(img_x)

        trann = transforms.ToPILImage()

        img_liver = torch.squeeze(img_liver)
        img_liver = img_liver.detach().cpu().numpy()
        img_liver = np.transpose(img_liver,[1,2,0])
        img_liver[img_liver>=0.5] = 255
        img_liver = img_liver.astype(np.uint8)           
        img_liver = trann(img_liver)
        img_liver = img_liver.convert('L')
        img_liver = img_liver.resize((h,w))
        img_liver = np.asarray(img_liver)
        


        img_tumor = torch.squeeze(img_tumor)
        img_tumor = img_tumor.detach().cpu().numpy()
        img_tumor = np.transpose(img_tumor,[1,2,0])
        img_tumor[img_tumor>=0.5] = 255
        img_tumor = img_tumor.astype(np.uint8)           
        img_tumor = trann(img_tumor)
        img_tumor = img_tumor.convert('L')
        img_tumor = img_tumor.resize((h,w))
        img_tumor = np.asarray(img_tumor)



        #肝脏后处理
        img_liver = img_liver.copy()
        img_liver[img_tumor==255] = 255
        kernel = np.ones((7,7),np.uint8) 
        img_liver[img_liver>=1] = 1
        img_liver = cv2.erode(img_liver,kernel)#腐蚀
        img_liver = cv2.dilate(img_liver,kernel,iterations = 1)#膨胀
        img_liver[img_liver>=1] = 255
        #肝脏后处理结束

        #肿瘤后处理
        img_tumor = img_tumor.copy()
        img_tumor[img_liver!=255] = 0
        #肿瘤后处理结束

        img_liver[img_liver>=1] = 1
        liver_target.append(img_liver)

        img_tumor[img_tumor>=1] = 1
        tumor_target.append(img_tumor)

    liver_target = np.asarray(liver_target)  
    liver_target = np.transpose(liver_target, [1,2,0])
    liver_nib = nib.Nifti1Image(liver_target, affine=np.eye(4))

    tumor_target = np.asarray(tumor_target)  
    tumor_target = np.transpose(tumor_target, [1,2,0])
    tumor_nib = nib.Nifti1Image(tumor_target, affine=np.eye(4))

    return liver_nib,tumor_nib



def run():
    liver_model_path = 'ckp_challenge_finally/liver_138.pth'
    tumor_model_path = 'ckp_challenge_finally/tumor_142.pth'

    liver_path = "D:/deep_learning/u_net_liver-test/liver"
    ls = os.listdir(liver_path)

    for i in range(len(ls)):
        nii_path = os.path.join(liver_path,ls[i],"liver.nii")
        liver_nib,tumor_nib = end_to_end(nii_path,liver_model_path,tumor_model_path)
        
        save_liver = os.path.join(liver_path,ls[i],"liver_seg.nii")
        save_tumor = os.path.join(liver_path,ls[i],"liver_nid.nii")

        nib.save(liver_nib,save_liver)
        nib.save(tumor_nib,save_tumor)


##处理一个病人10-15秒（GPU）
##处理一个病人169秒（CPU）

if __name__ == '__main__':
    
    time_start = time.time()

    liver_model_path = 'ckp_challenge_finally/liver_138.pth'
    tumor_model_path = 'ckp_challenge_finally/tumor_142.pth'

    
    nii_path = "liver/liver_1/liver.nii"

    liver_nib,tumor_nib = end_to_end(nii_path,liver_model_path,tumor_model_path)

    nib.save(liver_nib,"liver/liver_1/liver_seg.nii")
    nib.save(tumor_nib,"liver/liver_1/liver_nid.nii")
    
    time_end  = time.time()
    print(time_end-time_start)