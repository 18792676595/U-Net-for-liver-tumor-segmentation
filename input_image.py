# 输入图片，输出图片
import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from dice_loss import *
from eval_net import eval_net
from networks.cenet import CE_Net_
import cv2




import torch.nn as nn


def IoU(prediction, target):
    prediction = np.uint8(prediction)
    target = np.uint8(target)
    delta = 1e-10
    IoU = ((prediction * target).sum() + delta) / (prediction.sum() + target.sum() - (prediction * target).sum() + delta)

    return IoU

def Dice(prediction, target):
    prediction = np.uint8(prediction)
    target = np.uint8(target)
    delta = 1e-10
    return (2 * (prediction * target).sum() + delta) / (prediction.sum() + target.sum() + delta)




x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

y_transforms = transforms.ToTensor()


# 是否使用cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
def test_image():
    # model = Unet(3, 3)
    model = CE_Net_(3, 1).to(device)
    #model.load_state_dict(torch.load('ckp/CN_lits_transform_model_batch3.pth', map_location='cuda:0'))
    model.load_state_dict(torch.load('ckp_challenge_only/CE_LITS_and_challenge_model_epoch138.pth', map_location='cuda:0'))
    model.eval()

    root = "data/challenge_val/raw"
    label = "data/challenge_val/liver_target"
    save = "data/challenge_val/model_target"
    ls = os.listdir(root)
    for i in range(len(ls)):
        image_root = os.path.join(root,ls[i])
        img_x = Image.open(image_root)
        # img_x = img_x.convert('L')

        h,w = img_x.size
        img_x = img_x.resize((512,512))

        img_x = img_x.convert('RGB')
        img_x = x_transforms(img_x)
        img_x = img_x.to(device)
        img_x = torch.unsqueeze(img_x, 0)

        label_root = os.path.join(label,ls[i])
        img_y = Image.open(label_root)

        img_y = img_y.resize((512,512))
        # labels = labels.convert('L')

        img_y = img_y.convert('RGB')
        img_y = y_transforms(img_y)
        img_y = img_y.to(device)
        img_y = torch.unsqueeze(img_y, 0)

        out = model(img_x)
        print(IOU(out.to("cpu"), img_y.to("cpu")).item())

        trann = transforms.ToPILImage()
        out = torch.squeeze(out)
        out = out.detach().cpu().numpy()
        out = np.transpose(out,[1,2,0])
        out[out>=0.5] = 255
        out = out.astype(np.uint8)
        #out[out>=1] = 255
        out = trann(out)
        out = out.convert('L')
        out = out.resize((h,w))
        save_root = os.path.join(save,ls[i])
        out.save(save_root)

def Closed_operation():
    root = "data/challenge_val/model_target"
    save = "data/challenge_val/closedOP"
    ls = os.listdir(root)
    
    kernel = np.ones((5,5),np.uint8) 
    for i in range(len(ls)):
        image_root = os.path.join(root,ls[i])
        
        img = cv2.imread(image_root)
        img[img>=1] = 1
        #closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        closing = cv2.dilate(img,kernel,iterations = 2)
        closing[closing>=1] = 255
        save_root = os.path.join(save,ls[i])
        cv2.imwrite(save_root, closing)
        print(ls[i])
'''

def test_image():
    # model = Unet(3, 3)
    model = CE_Net_(3, 1).to(device)
    #model.load_state_dict(torch.load('ckp/CN_lits_transform_model_batch3.pth', map_location='cuda:0'))
    #model.load_state_dict(torch.load('ckp_challenge/CE_LITS_and_challenge_model_epoch123.pth', map_location='cuda:0'))
    model.load_state_dict(torch.load('ckp_challenge_only/CE_LITS_and_challenge_model_epoch138.pth', map_location='cuda:0'))
    
    model.eval()

    root = "data/challenge/raw"
    #label = "data/aug/val_target"
    save = "data/challenge/only_model_target"
    ls = os.listdir(root)
    for i in range(len(ls)):
        image_root = os.path.join(root,ls[i])
        img_x = Image.open(image_root)
        h,w = img_x.size
        img_x = img_x.resize((512,512))
        # img_x = img_x.convert('L')
        img_x = img_x.convert('RGB')
        img_x = x_transforms(img_x)
        img_x = img_x.to(device)
        img_x = torch.unsqueeze(img_x, 0)

        #label_root = os.path.join(label,ls[i])
        #img_y = Image.open(label_root)
        # labels = labels.convert('L')
        #img_y = img_y.convert('RGB')
        #img_y = y_transforms(img_y)
        #img_y = img_y.to(device)
        #img_y = torch.unsqueeze(img_y, 0)

        out = model(img_x)
        #print(IOU(out.to("cpu"), img_y.to("cpu")).item())

        trann = transforms.ToPILImage()
        out = torch.squeeze(out)
        out = out.detach().cpu().numpy()
        out = np.transpose(out,[1,2,0])
        out[out>=0.5] = 255
        out = out.astype(np.uint8)
        
        
        out = trann(out)
        out = out.convert('L')
        out = out.resize((h,w))
        save_root = os.path.join(save,ls[i])
        out.save(save_root)

def Closed_operation():
    root = "data/challenge/combine_only_model_target"
    save = "data/challenge/eroded_and_dilate7"
    ls = os.listdir(root)
    
    kernel = np.ones((7,7),np.uint8) 
    for i in range(len(ls)):
        image_root = os.path.join(root,ls[i])
        
        img = cv2.imread(image_root)
        img[img>=1] = 1
        #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        #closing = cv2.dilate(img,kernel,iterations = 1)#膨胀
        
        closing = cv2.erode(img,kernel)#腐蚀
        closing = cv2.dilate(closing,kernel,iterations = 1)#膨胀
        closing[closing>=1] = 255
        save_root = os.path.join(save,ls[i])
        cv2.imwrite(save_root, closing)
        print(ls[i])











def OP_IOU():
    #Pre_root = "data/challenge_val/model_target"
    Pre_root = "data/challenge_val/combine_only_model_target"
    tar_root = "data/challenge_val/liver_target"
    clos_root = "data/challenge_val/closedOP"

    ls = os.listdir(Pre_root)
    iou = 0
    n = 0
    for i in range(len(ls)):
        pre_image = os.path.join(Pre_root,ls[i])#0.91#0.92
        tar_image = os.path.join(tar_root,ls[i])
        clos_image = os.path.join(clos_root,ls[i])#0.93#0.79
        img_pre = cv2.imread(pre_image)
        img_pre[img_pre>=1] = 1
        img_label = cv2.imread(tar_image)
        img_label[img_label>=1] = 1
        #if IoU(img_pre,img_label)==1:#0.90
            #continue
        iou = iou + IoU(img_pre,img_label)
        n = n + 1
        print(IoU(img_pre,img_label))
    print(iou/n)


def combine_tumor_and_liver():
    liver_root = "data/challenge/only_model_target"
    tumor_root = "data/challenge/only_tumor_model_target"
    save_root = "data/challenge/combine_only_model_target"
    ls = os.listdir(liver_root)
    for n in range(len(ls)):
        liver = os.path.join(liver_root,ls[n])
        tumor = os.path.join(tumor_root,ls[n])
        save = os.path.join(save_root,ls[n])
        img_liver = cv2.imread(liver)
        img_tumor = cv2.imread(tumor)
        new_liver = img_liver
        new_liver[img_tumor==255] = 255
        cv2.imwrite(save,new_liver)
        print(ls[n])


                
        




if __name__ == '__main__':
    #test_image()
    Closed_operation()
    #OP_IOU()
    #combine_tumor_and_liver()