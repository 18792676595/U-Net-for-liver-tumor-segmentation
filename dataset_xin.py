import torch.utils.data as data
import PIL.Image as Image
import os

def make_dataset(root_liver,root_mask):
    imgs=[]
    #n=len(os.listdir(root_liver))//2
    n = len(os.listdir(root_liver))
    for i in range(n):
        if i==0:
            continue
        img=os.path.join(root_liver,"liver (%d).bmp"%i)
        mask=os.path.join(root_mask,"liver (%d).bmp"%i)
        imgs.append((img,mask))
    return imgs


class LiverDataset_xin(data.Dataset):
    def __init__(self, root_liver,root_mask, transform=None, target_transform=None):
        imgs = make_dataset(root_liver,root_mask)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        #img_x = img_x.resize((256,256))
        img_x = img_x.convert('L')#位深度24转换为8，转为灰度图
        img_y = Image.open(y_path)
        #img_y = img_y.resize((256,256))
        img_y = img_y.convert('L')#位深度24转换为8，转为灰度图
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)






