# -*- coding: utf-8 -*-
# @Author  : LG
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import os,sys,random,time
import argparse

'''
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1)) # b,c,h,w -> ,w
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
'''
# '''
class FocalLoss(nn.Module):
    """code refer to lpy"""
    def __init__(self): # 如果没有init方法,实例刚创建时就是一个简单的空的命名空间
        super(FocalLoss, self).__init__()
         
    def forward(self, outputs, targets):
        if outputs.dim()>2:
            outputs = outputs.view(outputs.size(0),outputs.size(1),-1)  # N,C,H,W => N,C,H*W
            outputs = outputs.transpose(1,2)    # N,C,H*W => N,H*W,C
            outputs = outputs.contiguous().view(-1,outputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(targets.size(0),targets.size(1),-1)  # N,C,H,W => N,C,H*W
            targets = targets.transpose(1,2)    # N,C,H*W => N,H*W,C
            targets = targets.contiguous().view(-1,targets.size(2))   # N,H*W,C => N*H*W,C
        # outputs = nn.Sigmoid()(outputs)
        # print(outputs.shape, targets.shape) # torch.Size([4, 2, 256, 256]) torch.Size([4, 2, 256, 256])
        # print(torch.unique(outputs), torch.unique(targets)) # torch.Size([4, 2, 256, 256]) torch.Size([4, 2, 256, 256])
        # alpha = 0.5 # 控制样本均衡 0.5相当于关掉此功能
        alpha = 0.25 # 论文中0.25 works best
        gamma = 2.0 # 控制难易程度学习 难学习的loss更大
        SMOOTH = 1e-4
        # outputs =  (outputs.squeeze(1)).squeeze(0) # DHW
        # targets = (targets.squeeze(1)).squeeze(0) #DHW
        channels = outputs.shape[1]
        channel_losses = []
        # print(outputs.shape, targets.shape)
        for c in range(channels):
            output = outputs[:, c] # N*H*W
            target = targets[:, c]
            # print(output.shape, target.shape)
            alpha_factor = torch.ones(target.shape).to(target.device) * alpha

            # 用两个where 来完成
            alpha_factor = torch.where(torch.eq(target, 1.), alpha_factor, 1. - alpha_factor)
            # print(alpha_factor.shape, alpha_factor.unique())

            focal_weight = torch.where( 
                torch.eq(target, 1.), 1. - output, output)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma) #αt(1 − pt)γ
            # print(focal_weight.shape)
            bce = -(target * torch.log(output + SMOOTH) +(1.0 - target) * torch.log(1.0 - output + SMOOTH))
            # print(bce.shape)
            # print(target[:10])
            # print(output[:10])
            # print(bce[:10])
            channel_focal_loss = focal_weight * bce
            # print(channel_focal_loss.shape)
            # print(type(channel_focal_loss.mean()), channel_focal_loss.mean())
            channel_losses.append(channel_focal_loss.mean())
        # return torch.stack(output).mean(dim=0, keepdim=True)
        # print(torch.stack(channel_losses))
        return torch.stack(channel_losses).mean(dim=0)
# '''
'''
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        print(class_mask)
        assert 1>4


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
'''
'''
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        print(target.shape)
        if self.alpha is not None:
            if self.alpha.type()!=input.type():
                print("="*20)
                self.alpha = self.alpha.type_as(input)
            at = self.alpha.gather(0,target.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
'''

if __name__ == "__main__":
    # start_time = time.time()
    # maxe = 0
    # for i in range(1000):
    #     x = torch.rand(10,2)*random.randint(1,10)
    #     # x = x.cuda()
    #     l = torch.rand(10).ge(0.4).int()
    #     # l = l.cuda()

    #     output0 = FocalLoss(gamma=2)(x,l)
    #     assert 1>3
    #     output1 = nn.CrossEntropyLoss()(x,l)
    #     a = output0.item()
    #     b = output1.item()
    #     if abs(a-b)>maxe: maxe = abs(a-b)
    # print('time:',time.time()-start_time,'max_error:',maxe)

    # tensor_0 = torch.arange(3, 12).view(3, 3)
    # index = torch.tensor([[2, 1, 0]])
    # tensor_1 = tensor_0.gather(0, index) # 8 7 5
    # print(tensor_1)
    # index = torch.tensor([[2, 1, 0]]).t()
    # tensor_1 = tensor_0.gather(1, index)
    # print(tensor_1)
    # index = torch.tensor([[0, 2], 
    #                   [1, 2]])
    # tensor_1 = tensor_0.gather(1, index)
    # print(tensor_1)
    # assert 1>2

    # start_time = time.time()
    # maxe = 0
    # for i in range(100):
    #     x = torch.rand(3,2,16,16)*random.randint(1,10) # b, c, h, w
    #     l = torch.rand(3,16,16)*2    # 1000 is classes_num
    #     l = l.long()

    #     output0 = FocalLoss(gamma=2)(x,l)
    #     assert 1>4
    #     output1 = nn.NLLLoss2d()(F.log_softmax(x),l)
    #     a = output0.data[0]
    #     b = output1.data[0]
    #     if abs(a-b)>maxe: maxe = abs(a-b)
    # print('time:',time.time()-start_time,'max_error:',maxe)

    loss = FocalLoss()
    inputs = torch.randn(4,2,512,512).sigmoid()
    targets = torch.rand(4,2,512,512).ge(0.4).int()
    # targets = torch.rand(10).ge(0.4).int()
    # bce = loss(targets, inputs)
    val = loss(inputs, targets)
    print(val)
    