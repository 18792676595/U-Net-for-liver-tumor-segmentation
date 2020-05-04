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
from dataset_lits import LiverDataset
from dice_loss import *
from eval_net import eval_net
from ounet import UNet
import pydicom
from networks.cenet import CE_Net_
from WL import *
from tensorboardX import SummaryWriter

writer_train = SummaryWriter("only_challenge/train")
writer_val = SummaryWriter("only_challenge/val")
wirter_all = SummaryWriter("only_challenge/all")

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
    model = CE_Net_(3,1).to(device)
    model.load_state_dict(torch.load('ckp/CN_lits_transform_model_batch2.pth',map_location='cuda:0'))
    batch_size = 20
    num_epochs = 50
    criterion = torch.nn.BCELoss()
    sm = torch.nn.Softmax(dim=1)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    root = "data/LITS_train"

    #save_data = "ckp_xin/f2model.txt"

    ls = os.listdir(root)


    train_iter = 0
    for n in range(len(ls)):
        if ls[n]=="batch1":
            continue
        if ls[n]=="batch2":
            continue
        root_liver = os.path.join(root,ls[n],"raw")
        root_mask = os.path.join(root,ls[n],"liver_target")
        liver_dataset = LiverDataset(root_liver,root_mask,transform=trainx_transforms,target_transform=trainy_transforms)
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
            
            liver_dataset_test = LiverDataset("data/LITS_val/raw","data/LITS_val/liver_target", transform=x_transforms,target_transform=y_transforms)
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

        torch.save(model.state_dict(),'ckp/CN_lits_transform_model_batch%d.pth'%(n+1))



    

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
    model = CE_Net_(1,1).to(device)
    model.load_state_dict(torch.load('ckp_xin/CN_lits1_model.pth',map_location='cpu'))
    model.eval()

 
    img_x = Image.open("data/aug/32.bmp")
    #img_x = img_x.convert('L')
    img_x = img_x.convert('RGB')


    img_x = x_transforms(img_x)
    img_x = torch.unsqueeze(img_x,0)


    labels = Image.open("data/aug/32_mask.bmp")
    #labels = labels.convert('L')
    img_y = img_y.convert('RGB')
    labels = y_transforms(labels)
    labels = torch.unsqueeze(labels,0)


    out = model(img_x)
    print(IOU(out.to("cpu"),labels.to("cpu")).item())

    trann = transforms.ToPILImage()
    out = torch.squeeze(out)
    out = trann(out)
    out.save("data/aug/32_maskfd1.bmp")
    







import torch.onnx
import torchvision.models as models 
writer_graph = SummaryWriter("run_CN_LITS_1/graph")

#得到模型图
def get_model_graph():

    #model = Unet(3,1).to(device)
    #model = CE_Net_(1,1)
    model = CE_Net_(3,1)
    '''
    model.load_state_dict(torch.load('ckp_xin/f1model.pth',map_location='cpu'))
    model.eval()
    '''
    img = Image.open("data/aug/32.bmp")
    img = img.convert('L')
    img = x_transforms(img)
    img = torch.unsqueeze(img,0)
    
    input_data = Variable(torch.randn(1,1,512,512))
    
    #print(input_data.size())
    writer_graph.add_graph(model,input_to_model=input_data)
    



    '''
    resnet18 = models.resnet18(False) 
    dummy_input = torch.rand(6, 3, 224, 224) 
    writer_graph.add_graph(resnet18, dummy_input) 
    '''

    

    
   


def trainmore():
    model = CE_Net_(1,1).to(device)
    batch_size = 20
    num_epochs = 50
    criterion = torch.nn.BCELoss()
    #sm = torch.nn.Softmax(dim=1)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    train_iter = 0
    model.load_state_dict(torch.load('ckp_xin/CN_lits1_model.pth',map_location='cuda:0'))
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

            print("epoch %d ave_loss:%f,ave_acc:%f" % (epoch+1, epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),tot_acc/((dt_size - 1) / dataloaders.batch_size + 1)))

            writer_train.add_scalar("train_loss",epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),train_iter)
            writer_train.add_scalar("train_acc",tot_acc/((dt_size - 1) / dataloaders.batch_size + 1),train_iter)

            
            liver_dataset_test = LiverDataset_xin("data/val_xin/bmp","data/val_xin/mask_bw", transform=x_transforms,target_transform=y_transforms)
            dataloaders_test = DataLoader(liver_dataset_test, batch_size=1)
            criterion = torch.nn.BCELoss()
            model = model.eval()
            iou,loss = test_model(model,criterion,dataloaders_test)

            writer_val.add_scalar("val_loss",loss,train_iter)
            writer_val.add_scalar("val_acc",iou,train_iter)

            wirter_all.add_scalars("loss",{'train_loss':epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),'val_loss':loss},train_iter)
            wirter_all.add_scalars("acc",{'train_acc':tot_acc/((dt_size - 1) / dataloaders.batch_size + 1),'val_acc':iou},train_iter)

    torch.save(model.state_dict(),'ckp_xin/CE_LITS_and_hospital_model.pth')


def train_more_challenge():
    model = CE_Net_(3,1).to(device)
    batch_size = 16
    num_epochs = 500
    criterion = torch.nn.BCELoss()
    #sm = torch.nn.Softmax(dim=1)
    optimizer = optim.Adam(model.parameters(),lr=1e-4)
    train_iter = 0
    #model.load_state_dict(torch.load('ckp/CN_lits_transform_model_batch3.pth',map_location='cuda:0'))
    root = "data/challenge_train"

    #save_data = "ckp_xin/f2model.txt"

    ls = os.listdir(root)


    train_iter = 0
    for n in range(len(ls)):
        root_liver = os.path.join(root,ls[n],"raw")
        root_mask = os.path.join(root,ls[n],"liver_target")
        liver_dataset = LiverDataset(root_liver,root_mask,transform=trainx_transforms,target_transform=trainy_transforms)
        dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        #model = train_model(model, criterion, optimizer, dataloaders)
        for epoch in range(num_epochs):
            print('Epoch {}/{}/{}'.format(epoch+1, num_epochs,n+1))
            print('-' * 10)
            train_iter += 1

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

            print("epoch %d ave_loss:%f,ave_acc:%f" % (epoch+1, epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),tot_acc/((dt_size - 1) / dataloaders.batch_size + 1)))

            writer_train.add_scalar("train_loss",epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),train_iter)
            writer_train.add_scalar("train_acc",tot_acc/((dt_size - 1) / dataloaders.batch_size + 1),train_iter)

            
            liver_dataset_test = LiverDataset("data/challenge_val/raw","data/challenge_val/liver_target", transform=x_transforms,target_transform=y_transforms)
            dataloaders_test = DataLoader(liver_dataset_test, batch_size=1)
            criterion = torch.nn.BCELoss()
            model = model.eval()
            iou,loss = test_model(model,criterion,dataloaders_test)

            writer_val.add_scalar("val_loss",loss,train_iter)
            writer_val.add_scalar("val_acc",iou,train_iter)

            wirter_all.add_scalars("loss",{'train_loss':epoch_loss/((dt_size - 1) / dataloaders.batch_size + 1),'val_loss':loss},train_iter)
            wirter_all.add_scalars("acc",{'train_acc':tot_acc/((dt_size - 1) / dataloaders.batch_size + 1),'val_acc':iou},train_iter)

            torch.save(model.state_dict(),'ckp_challenge_only/CE_LITS_and_challenge_model_epoch%d.pth'%(epoch+1))