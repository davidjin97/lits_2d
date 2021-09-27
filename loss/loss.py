import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss for LITS dataset, which has liver mask and tumor mask
    """
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        inputs = torch.sigmoid(input)
        num = target.size(0)
        input_1 = inputs[:,0,:,:,:]
        input_2 = inputs[:,1,:,:,:]

        target_1 = target[:,0,:,:,:]
        target_2 = target[:,1,:,:,:]

        input_1 = input_1.view(num, -1)
        target_1 = target_1.view(num, -1)

        input_2 = input_2.view(num, -1)
        target_2 = target_2.view(num, -1)

        intersection_1 = (input_1 * target_1)
        intersection_2 = (input_2 * target_2)

        dice_1 = (2. * intersection_1.sum(1) + smooth) / (input_1.sum(1) + target_1.sum(1) + smooth)

        dice_2 = (2. * intersection_2.sum(1) + smooth) / (input_2.sum(1) + target_2.sum(1) + smooth)

        dice_1 = 1 - dice_1.sum() / num
        dice_2 = 1 - dice_2.sum() / num

        dice = dice_1*0.4+dice_2*0.6
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0) # batch_size
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha=0.5, beta=0.5):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.beta = beta
        self.dice = SoftDiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)



class Active_Contour_Loss(nn.Module):
    def __init__(self):
        super(Active_Contour_Loss, self).__init__()

    def forward(self, y_true, y_pred): 
        
        x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 水平和垂直方向 HW
        y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

        delta_x = x[:,:,1:,:-2]**2
        delta_y = y[:,:,:-2,1:]**2
        delta_u = torch.abs(delta_x + delta_y) 

        lenth = torch.mean(torch.sqrt(delta_u + 0.00000001)) # equ.(11) in the paper

        """
        region term
        """

        C_1 = torch.ones((256, 256))
        C_2 = torch.zeros((256, 256))

        region_in = torch.abs(torch.mean( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
        region_out = torch.abs(torch.mean( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

        lambdaP = 1 # lambda parameter could be various.
        mu = 1 # mu parameter could be various.
        
        return lenth + lambdaP * (mu * region_in + region_out) 

class DiceLoss(nn.Module): 
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, logits, targets):
        batch_size = targets.size(0)
        log_prob = torch.sigmoid(logits)                                                                                                                                            
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
        dice_score = 2. * intersection / ((logits + targets).sum(-1) + self.epsilon)
        # dice_score = 1 - dice_score.sum() / batch_size
        return torch.mean(1. - dice_score)

class TverskyLoss(nn.Module):
    def __init__(self, alpha = 0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
    # def tversky(self, y_true, y_pred):
    #     y_true_pos = K.flatten(y_true)
    #     y_pred_pos = K.flatten(y_pred)
    #     true_pos = K.sum(y_true_pos * y_pred_pos)
    #     false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    #     false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    #     alpha = 0.7
    #     return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

    def forward(self, y_true,y_pred): # 传入单张
        y_true = (y_true.squeeze(1)).squeeze(0) # DHW 仅针对batchsize ==  1的情况下
        y_pred = (y_pred.squeeze(1)).squeeze(0) # DHW
        
        # seqDepth = y_pred.shape[0]
        # losses = []
        
        # for j in range(seqDepth):
        tp = (y_true * y_pred).sum() # 交集
        
        fp = ((1-y_true) * y_pred).sum()
        fn = (y_true * (1-y_pred)).sum()
        tversky = tp + SMOOTH / (tp + self.alpha*fp + self.beta*fn + SMOOTH) 

        tversky_loss = 1 - tversky
        
        # pt_1 = tversky(y_true, y_pred)
        # gamma = 0.75
        # focal_tversky = torch.pow((1-pt_1), gamma)
        
        return tversky_loss

class CrossEntropyLoss:
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self._backend_loss = nn.CrossEntropyLoss(self.weight,
                                                 ignore_index=self.ignore_index,
                                                 reduction=self.reduction)

    def __call__(self, input, target, scale=[0.4, 1.]):
        '''
        :param input: [batch_size,c,h,w]
        :param target: [batch_size,h,w]
        :param scale: [...]
        :return: loss
        '''
        if isinstance(input, tuple) and (scale is not None):
            loss = 0
            for i, inp in enumerate(input):
                loss += scale[i] * self._backend_loss(inp, target)
            return loss
        else:
            return self._backend_loss(input, target)


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=0.001):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        depth = x.size()[2]
        h_x = x.size()[3]
        w_x = x.size()[4]
        
        count_h = self._tensor_size(x[:,:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,:,1:])
        h_tv = torch.pow((x[:,:,:,1:,:]-x[:,:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,:,1:]-x[:,:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/(batch_size + depth)

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]*t.size()[4]




if __name__ == "__main__":
    # loss = TverskyLoss()
    loss = FocalLoss()
    inputs = torch.randn(4,2, 512,512)
    targets = torch.randn(4,2,512, 512)
    # bce = loss(targets, inputs)
    val = loss(inputs, targets)
    print(val)
    