import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from torch import autograd, optim
from torchvision.transforms import transforms
from unet import Unet
#from dataset import LiverDataset
from dataset_xin import LiverDataset_xin
#from dataset_dicom import LiverDataset
from dice_loss import *
from eval_net import eval_net
from ounet import UNet
import pydicom
from networks.cenet import CE_Net_
from WL import *
from tensorboardX import SummaryWriter
from FCN_PP_VGG16 import MUNET
writer_train = SummaryWriter("rundc3_spp/train")
writer_val = SummaryWriter("rundc3_spp/val")
wirter_all = SummaryWriter("rundc3_spp/all")

# 是否使用cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

trainx_transforms = transforms.Compose([
    #transforms.Resize((256,256)),
    #transforms.RandomHorizontalFlip(),#依概率p水平翻转
    #transforms.RandomVerticalFlip(),#依概率p垂直翻转
    #transforms.RandomCrop(100),   
    transforms.ToTensor(),   
    transforms.Normalize([0.5], [0.5])
])

# mask只需要转换为tensor
trainy_transforms = transforms.Compose([
    #transforms.Resize((256,256)),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(100),   
    transforms.ToTensor(),   
])




x_transforms = transforms.Compose([
    transforms.ToTensor(),   
    transforms.Normalize([0.5], [0.5])
])

y_transforms = transforms.ToTensor()



def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        tot_acc = 0
        step = 0
        
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            train_acc = dice_coeff(outputs.to("cpu"),labels.to("cpu")).item()
            tot_acc += train_acc
            print("%d/%d,train_loss:%f,train_acc:%f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item(),train_acc))
            
        print("epoch %d ave_train_loss:%f,ave_train_acc:%f" % (epoch, epoch_loss/((dt_size - 1) / dataload.batch_size + 1),tot_acc/((dt_size - 1) / dataload.batch_size + 1)))
    #torch.save(model.state_dict(), 'ckp/weights_%d.pth' % epoch+1)
    #torch.save(model.state_dict(),'ckp_xin/n3model.pth')
    return model

def test_model(model,criterion,dataload):
    '''
    if dataload is not None:
        valid_loss = 0
        valid_acc = 0
        print(int(len(dataload)))
        for x,y in dataload:
            #inputs = x.to(device)
            #labels = y.to(device)
            out = model(x)
            loss = criterion(out,y)
            print(loss.item())
            valid_loss += loss.item()
            #valid_acc += get_acc(out,y)
            
        #epoch_str = ("Valid Loss: %f, Valid Acc: %f, " % valid_loss / len(dataload),valid_acc / len(dataload))
        epoch_str = ("Valid Loss: %f " % valid_loss / int(len(dataload)))
        
    else:
        epoch_str = ("test_dataload is none")
    
    print(epoch_str)
    '''
    iou,loss = eval_net(model,criterion,dataload,True)
    #dice,loss = eval_net(model,criterion,dataload)
    print("ave_test_iou:{},ave_test_loss:{}".format(iou,loss))

    '''
    save_data = "ckp_xin/test/caoying.txt"
    with open(save_data,'a+') as fw:
        fw.write("ave_test_dice:{},ave_test_loss:{}".format(dice,loss))
        fw.write("\n")
    '''


    return iou,loss
    












#训练模型
def train():
    model = Unet(3, 3).to(device)
    batch_size = 3
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-5)
    liver_dataset = LiverDataset("data/train_xin/liver_bmp","data/train_xin/mask_bw",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, criterion, optimizer, dataloaders)
    #torch.save(model.state_dict(),'ckp/model.pth')




#循环读取文件夹进行学习
def train_cir():
    #model = Unet(1,1).to(device)
    #model = CE_Net_(3,1).to(device)
    model = MUNET(1,1).to(device)
    batch_size = 5
    num_epochs = 100
    criterion = torch.nn.BCELoss()
    #sm = torch.nn.Softmax(dim=1)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)

    root = "data/tri"

    #save_data = "ckp_xin/f2model.txt"

    ls = os.listdir(root)


    train_iter = 0
    for n in range(len(ls)):
        root_liver = os.path.join(root,ls[n],"bmp")
        root_mask = os.path.join(root,ls[n],"mask_bw")
        liver_dataset = LiverDataset_xin(root_liver,root_mask,transform=trainx_transforms,target_transform=trainy_transforms)
        dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        #model = train_model(model, criterion, optimizer, dataloaders)
        for epoch in range(num_epochs):
            print('Epoch {}/{}/{}'.format(epoch+1, num_epochs,n+1))
            print('-' * 10)
            train_iter += 1
            '''
            with open(save_data,'a+') as fw:
                fw.write('Epoch {}/{}/{}'.format(epoch+1, num_epochs,n+1))
                fw.write('\n')
                fw.write('-' * 10)
                fw.write('\n')

            '''
            dt_size = len(dataloaders.dataset)
            epoch_loss = 0
            tot_acc = 0
            step = 0

            model = model.train()

            for x, y in dataloaders:
                step += 1
                
                inputs = x.to(device)
                labels = y.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

                train_acc = IOU(outputs.to("cpu"),labels.to("cpu")).item()


                tot_acc += train_acc
                print("%d/%d,train_loss:%f,train_acc:%f" % (step, (dt_size - 1) // dataloaders.batch_size + 1, loss.item(),train_acc))

                '''
                with open(save_data,'a+') as fw:
                    fw.write("%d/%d,train_loss:%f,train_acc:%f" % (step, (dt_size - 1) // dataloaders.batch_size + 1, loss.item(),train_acc))
                    fw.write("\n")
                '''
            print("epoch %d ave_loss:%f,ave_acc:%f" % (epoch+1, epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),tot_acc/((dt_size - 1) / dataloaders.batch_size + 1)))

            writer_train.add_scalar("train_loss",epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),train_iter)
            writer_train.add_scalar("train_acc",tot_acc/((dt_size - 1) / dataloaders.batch_size + 1),train_iter)

            '''
            with open(save_data,'a+') as fw:
                fw.write("epoch %d ave_loss:%f,ave_acc:%f" % (epoch+1, epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),tot_acc/((dt_size - 1) / dataloaders.batch_size + 1)))
                fw.write("\n")
            '''
            
            liver_dataset_test = LiverDataset_xin("data/val_xin/bmp","data/val_xin/mask_bw", transform=x_transforms,target_transform=y_transforms)
            dataloaders_test = DataLoader(liver_dataset_test, batch_size=1)
            criterion = torch.nn.BCELoss()
            model = model.eval()
            iou,loss = test_model(model,criterion,dataloaders_test)

            '''
            with open(save_data,'a+') as fw:
                 fw.write("ave_test_dice:{},ave_test_loss:{}".format(dice,loss))
                 fw.write("\n")
            '''
            writer_val.add_scalar("val_loss",loss,train_iter)
            writer_val.add_scalar("val_acc",iou,train_iter)

            wirter_all.add_scalars("loss",{'train_loss':epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),'val_loss':loss},train_iter)
            wirter_all.add_scalars("acc",{'train_acc':tot_acc/((dt_size - 1) / dataloaders.batch_size + 1),'val_acc':iou},train_iter)
            
            if((epoch+1)%30==0):
                torch.save(model.state_dict(),'ckp_xin/sppepoch/unetspphospital_batch5_epoch{}_{}model.pth'.format((epoch+1),(n+1)))



    

'''
#显示模型的输出结果
def test():
    model = Unet(3, 1)
    model.load_state_dict(torch.load('ckp/model.pth',map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    import matplotlib.pyplot as plt
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y=model(x)
            img_y=torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(0.01)
        plt.show()
'''

#显示测试准确率
def test_acc():
    model = UNet(3, 1)
    model.load_state_dict(torch.load('ckp_xin/f3model.pth'))
    liver_dataset = LiverDataset("data/va/ZHOU YING JU002/liver_bmp","data/va/ZHOU YING JU002/mask_bw", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    criterion = torch.nn.BCELoss()
    model.eval()
    test_model(model,criterion,dataloaders)






#输入图片，输出图片
import torch.utils.data as data
import PIL.Image as Image
import os
def test_image():
    #model = Unet(3, 3)
    model = Unet(1,1)
    model.load_state_dict(torch.load('ckp_xin/fd3model.pth',map_location='cpu'))
    model.eval()

    '''
    img = Image.open("data/aug/24.bmp")
    img = x_transforms(img)
    img = torch.unsqueeze(img,0)
    '''
    #img_x = pydicom.dcmread("data/aug/32.dcm")
    #img_x = WL(img_x,150,300)
    #img_x = Image.fromarray(img_x)
    img_x = Image.open("data/aug/76.bmp")
    img_x = img_x.convert('L')
    #img_x.save('data/aug/32dtp.bmp')
    #img_x.show()

    img_x = x_transforms(img_x)
    img_x = torch.unsqueeze(img_x,0)


    labels = Image.open("data/aug/76_mask.bmp")
    labels = labels.convert('L')
    labels = y_transforms(labels)
    labels = torch.unsqueeze(labels,0)


    out = model(img_x)
    print(IOU(out.to("cpu"),labels.to("cpu")).item())
    '''
    img_mask = Image.open("data/aug/166_mask.png")
    img_mask = y_transforms(img_mask)
    img_mask = torch.unsqueeze(img_mask,0)

    out = model(img)
    dice = dice_coeff(out,img_mask)
    print(dice.detach().numpy())
    '''

    trann = transforms.ToPILImage()
    out = torch.squeeze(out)
    out = trann(out)
    out.save("data/aug/76_maskfd3.bmp")
    







import torch.onnx
import torchvision.models as models 
writer_graph = SummaryWriter("runs/graph")

#得到模型图
def get_model_graph():

    model = Unet(3,1).to(device)
    '''
    model.load_state_dict(torch.load('ckp_xin/f1model.pth',map_location='cpu'))
    model.eval()
    img = Image.open("data/aug/1.bmp")
    img = x_transforms(img)
    img = torch.unsqueeze(img,0)
    '''
    input_data = Variable(torch.randn(1,3,352,352)).to(device)
    
    #print(input_data.size())
    writer_graph.add_graph(model,input_to_model=input_data)
    



    '''
    resnet18 = models.resnet18(False) 
    dummy_input = torch.rand(6, 3, 224, 224) 
    writer_graph.add_graph(resnet18, dummy_input) 
    '''

    

    
   
