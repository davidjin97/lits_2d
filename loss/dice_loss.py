import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module): 
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        batch_size = targets.size(0)
        # log_prob = torch.sigmoid(outputs)                                                                                                                                            
        outputs = outputs.view(batch_size, -1).type(torch.FloatTensor) # 4, 2*512*512
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (outputs * targets).sum(-1)
        dice_score = (2. * intersection + self.epsilon) / ((outputs + targets).sum(-1) + self.epsilon)
        return torch.mean(1. - dice_score)

if __name__ == "__main__":
    loss = DiceLoss()
    inputs = torch.randn(4,2,512,512).sigmoid()
    targets = torch.rand(4,2,512,512).ge(0.4).int()
    # targets = torch.rand(10).ge(0.4).int()
    # bce = loss(targets, inputs)
    val = loss(inputs, targets)
    print(val)