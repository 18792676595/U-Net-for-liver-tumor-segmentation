# 输入图片，输出图片
import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
import glob
import nibabel as nib
import scipy.io as sio
import os
import imageio
import cv2




def test_image():
    # model = Unet(3, 3)
    model = CE_Net_(1, 1).to(device)
    model.load_state_dict(torch.load('ckp/CN_lits_transform_model_batch2.pth', map_location='cpu'))
    model.eval()

    filenames = glob.glob('batch1/*volume*')

    volumes = []
    for name in filenames:
        print(name)
        volume = nib.load(name).get_fdata()
        volume = np.transpose(volume, [2, 0, 1])
        volumes.append(volume)

    for p, volume in enumerate(volumes):
        print(filenames[p])
        for i in range(volume.shape[0]):
            a = volume[i]
            dcm_img = Image.fromarray(np.int16(np.array(volume[i])))
            dcm_img = dcm_img.convert('RGB')
            dcm_img.save(os.path.join('raw', filenames[p][14:-4] + '_{:04d}.bmp'.format(i)))

            img_x = dcm_img
            img_x = x_transforms(img_x)
            img_x = torch.unsqueeze(img_x, 0)
            out = model(img_x)
            trann = transforms.ToPILImage()
            out = torch.squeeze(out)
            out = trann(out)
            out.save(os.path.join('output', filenames[p][14:-4]+'_{:04d}.bmp'.format(i)))

    filenames = glob.glob('batch1/*seg*')

    volumes = []
    for name in filenames:
        print(name)
        volume = nib.load(name).get_fdata()
        volume = np.transpose(volume, [2, 0, 1])
        # volume[volume>=1] = 255
        volume[volume == 1] = 0
        volume[volume == 2] = 255
        volumes.append(volume)

    for p, volume in enumerate(volumes):
        print(filenames[p])
        for i in range(volume.shape[0]):
            dcm_img = Image.fromarray(np.uint8(np.array(volume[i])))
            dcm_img = dcm_img.convert('RGB')
            dcm_img.save(os.path.join('tumor_target', filenames[p][20:-4] + '_{:04d}.bmp'.format(i)))
            print(np.unique(np.array(volume[i])))

    volumes = []
    for name in filenames:
        print(name)
        volume = nib.load(name).get_fdata()
        volume = np.transpose(volume, [2, 0, 1])
        volume[volume>=1] = 255
        #volume[volume == 1] = 0
        #volume[volume == 2] = 255
        volumes.append(volume)

    for p, volume in enumerate(volumes):
        print(filenames[p])
        for i in range(volume.shape[0]):
            dcm_img = Image.fromarray(np.uint8(np.array(volume[i])))
            dcm_img = dcm_img.convert('RGB')
            dcm_img.save(os.path.join('liver_target', filenames[p][20:-4] + '_{:04d}.bmp'.format(i)))
            print(np.unique(np.array(volume[i])))




    labels = Image.open("data/aug/32_mask.bmp")
    # labels = labels.convert('L')
    img_y = img_y.convert('RGB')
    labels = y_transforms(labels)
    labels = torch.unsqueeze(labels, 0)

    out = model(img_x)
    print(IOU(out.to("cpu"), labels.to("cpu")).item())

