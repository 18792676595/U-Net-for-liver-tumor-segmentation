import torch
from torch.autograd import Function, Variable
import torch.nn as nn
class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        
        sm = nn.Softmax(dim=1)
        input = sm(input)
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        #调试
        a = torch.sum(input)
        b = torch.sum(target)
        c = input.view(-1)

        t = (2 * self.inter.float() + eps) / self.union.float()

        

        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)




def IOU(input,target):
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    eps = 0.0001
    for i,c in enumerate(zip(input,target)):

        #test
        a = input.view(-1)
        b = target.view(-1)



        inter = torch.dot(input.view(-1),target.view(-1))
        union = torch.sum(input) + torch.sum(target) + eps

        iou = (inter.float()+eps) / (union.float()-inter.float())

        s += iou

    return s / (i+1) 


#def IOU2(input,target):
    