import torch
import torch.nn.functional as F
import numpy as np
from dice_loss import dice_coeff
import os
from dice_loss import IOU




def eval_net(net, criterion,dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    loss_total = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        '''
        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        '''
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = torch.unsqueeze(mask_pred,0)
        mask_pred = (mask_pred > 0.5).float()

        #tot += dice_coeff(mask_pred, true_mask).item()
        tot += IOU(mask_pred, true_mask).item()
        #print(dice_coeff(mask_pred, true_mask).item())

        loss = criterion(mask_pred,true_mask)
        loss_total += loss.item()

        #print("test_dice:{},test_loss:{}".format(IOU(mask_pred, true_mask).item(),loss))

        #pr = mask_pred.cpu().detach().numpy()
        #a = true_mask.cpu().detach().numpy()

        #tot+=Dice(pr,ta)

        '''
        print("No:{},test_dice:{},test_loss:{}".format(i+1,dice_coeff(mask_pred, true_mask).item(),loss.item()))
        save_data = "ckp_xin/test/caoying.txt"
        with open(save_data,'a+') as fw:
            fw.write("No:{},test_dice:{},test_loss:{}".format(i+1,dice_coeff(mask_pred, true_mask).item(),loss.item()))
            fw.write("\n")
        '''
    return tot / (i + 1),loss_total/(i+1)



    
