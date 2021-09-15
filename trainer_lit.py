import os
from glob import glob
import time
import datetime
import shutil
import logging
import numpy as np
import cv2
import skimage.io
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from net_builder import build_net
# from loss.loss impojrt FocalLoss2d
from evaluate.metric import *
# from utils.torchsummary import summary

from dataset.lits.dataset import Dataset
from dataset.lits.lits_2d import LitsDataset

logger = logging.getLogger('global')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SegTrainer(object):
    def __init__(self, opt):
        super(SegTrainer, self).__init__()
        self.opt = opt
        self.initialize()

    def initialize(self):
        self._build_net()
        self._get_parameter_number()
        self._prepare_dataset()
        if not self.opt.evaluate:
            self._build_optimizer()
    
    def _build_net(self):
        self.net = build_net(self.opt.net_name)(**self.opt.model).to(self.opt.rank)
        # 这边可以添加summary，格式化网络 -- todo

        if not self.opt.evaluate:
            if self.opt.bn == 'sync':
                try:
                    import apex # not install yet
                    self.net = apex.parallel.convert_syncbn_model(self.net)
                except:
                    logger.info('not install apex. thus no sync bn')
            elif self.opt.bn == 'freeze':
                self.net = self.net.apply(freeze_bn)

        # 使用DDP       
        self.net = DDP(self.net,device_ids=[self.opt.rank])

        if self.opt.evaluate:
            self.load_state_keywise(self.opt.resume_model)
            logger.info('Load resume model from {}'.format(self.opt.resume_model))
        elif self.opt.pretrain_model == '':
            logger.info('Initial a new model...')
        else:
            if os.path.isfile(self.opt.pretrain_model):
                self.load_state_keywise(self.opt.pretrain_model)
                logger.info('Load pretrain model from {}'.format(self.opt.pretrain_model))
            else:
                logger.error('Can not find the specific model %s, initial a new model...', self.opt.pretrain_model)


        logger.info('Build model done.')

    def _get_parameter_number(self):
        total_num = sum(p.numel() for p in self.net.parameters())
        trainable_num = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logger.info(f'Total net parameters are {total_num//(1e6)}M, and weight file size is {trainable_num*4//(1024**2)}MB')

    def _prepare_dataset(self):
        if not self.opt.evaluate:
            img_paths = glob('/home/jzw/data/LiTS/LITS17/train_image_352*352/*')
            # mask_paths = glob('/home/jzw/data/LiTS/LITS17/train_mask_224*224/*')
            if self.opt.debug:
                img_paths = img_paths[:20]
                # mask_paths = mask_paths[:20]
            train_img_paths, val_img_paths= \
                train_test_split(img_paths, test_size=0.3, random_state=self.opt.manualSeed)

            train_dataset = LitsDataset(self.opt, train_img_paths)
            self.n_train_img = len(train_dataset)
            self.max_iter = self.n_train_img * self.opt.train_epoch // self.opt.batch_size // self.opt.world_size
            train_sampler = DistributedSampler(train_dataset)
            self.train_loader = DataLoader(train_dataset, shuffle=False, num_workers=0, batch_size=self.opt.batch_size,
                                           pin_memory=True, sampler=train_sampler)
            logger.info('train with {} pair images'.format(self.n_train_img))

            # self.val_loader = []
            # self.opt.val_dir = self.opt.get('val_dir', '')
            # # if self.opt.val_dir != '':
            #     # val_dataset = Lits_DataSet(self.opt.test_list,[48, 256, 256],1,self.opt.test_dir)
            val_dataset = LitsDataset(self.opt, val_img_paths)
            self.n_val_img = len(val_dataset)
            val_sampler = DistributedSampler(val_dataset)
            self.val_loader = DataLoader(val_dataset, shuffle=False, num_workers=0, batch_size=1,
                                            pin_memory=True, sampler=val_sampler)
            logger.info('val with {} pair images'.format(self.n_val_img))

        # test_dataset = Lits_DataSet(self.opt.test_list,[48, 256, 256],1,self.opt.test_dir)
        # self.n_test_img = len(test_dataset)
        # test_sampler = DistributedSampler(test_dataset)
        # self.test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0, batch_size=1,
        #                               pin_memory=True, sampler=test_sampler)
        # logger.info('test with {} pair images'.format(self.n_test_img))
        # logger.info('Build dataset done.')

    def _build_optimizer(self):
        # construct optimizer
        if self.opt.optimizer.type == 'SGD':
            self.optimizer = torch.optim.SGD(self.net.parameters(), self.opt.lr_scheduler.base_lr,
                                             momentum=0.9, weight_decay=0.0005)
        elif self.opt.optimizer.type == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), self.opt.lr_scheduler.base_lr)
        else: # == 'Adam'
            self.optimizer = torch.optim.Adam(self.net.parameters(), self.opt.lr_scheduler.base_lr, amsgrad=True)

        # construct lr_scheduler to adjust learning rate
        if self.opt.lr_scheduler.type == 'STEP':
            self.lr_scheduler = StepLR(optimizer=self.optimizer,step_size=self.opt.lr_scheduler.step_size,gamma=self.opt.lr_scheduler.gamma)


        # construct loss function
        if self.opt.loss.type == 'multi_scale':
            self.loss_function = MultiScaleLoss(**self.opt.loss.kwargs)
        elif self.opt.loss.type == 'focal2d':
            self.loss_function = FocalLoss2d(**self.opt.loss.kwargs)
        elif self.opt.loss.type == 'focal3d':
            self.loss_function = FocalLoss3d()
        elif self.opt.loss.type == 'kl':
            self.loss_function = nn.KLDivLoss()
        elif self.opt.loss.type == 'l1':
            self.loss_function = nn.L1Loss()
        elif self.opt.loss.type == 'ce':
            self.loss_function = nn.CrossEntropyLoss()
        # bce的shape需要一致
        elif self.opt.loss.type == 'bce':
            self.loss_function = nn.BCELoss()
        elif self.opt.loss.type == 'bcel':
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            logger.error('incorrect loss type {}'.format(self.opt.loss.type))

    def train(self):
        self.net.train()
        if self.opt.rank == 0:
            if not self.opt.log_directory:
                self.opt.log_directory = os.makedirs(self.opt.log_directory)
            self.writer = SummaryWriter(self.opt.log_directory)

        train_start_time = time.time()
        losses = AverageMeter()
        # ious = AverageMeter()
        # dices_1s = AverageMeter()
        # dices_2s = AverageMeter()
        for epoch in range(self.opt.train_epoch):
            epoch_iters = len(self.train_loader)
            for iter_train, (image, mask) in enumerate(self.train_loader):
                # print(image.shape, mask.shape) # torch.Size([b, 1, 64, 128, 160]) torch.Size([b, 2, 64, 128, 160])
                image = image.to(self.opt.rank)
                mask = mask.to(self.opt.rank)

                output = self.net(image) # [2, 2, 352, 352]

                loss = self.loss_function(output, mask)
                # iou = iou_score(output, target) 
                # dice_1 = dice_coef(output, target)[0]
                # dice_2 = dice_coef(output, target)[1]

                losses.update(loss.item(), image.shape[0])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                logger.info(f'Training Epoch: {epoch+1}/{self.opt.train_epoch},iter: {iter_train+1}/{epoch_iters}, the loss is {loss.item()}')
                if self.opt.rank == 0:
                    self.writer.add_scalar(f'Loss/train_iter', loss.item(), iter_train + epoch * len(self.train_loader))
            # self.lr_scheduler.step()

            logger.info(f'Start evalute at Epoch: {epoch+1}/{self.opt.train_epoch}')
            self.val(epoch)
            self.net.train()

            if self.opt.rank == 0 and (epoch+1)%self.opt.save_every_epoch == 0:
                self.save_checkpoint({'epoch': self.opt.train_epoch,
                                    'arch': self.opt.net_name,
                                    'state_dict': self.net.state_dict(),
                                    }, f'epoch_{epoch+1}_model.pth')
        logger.info(f"Finish training at {datetime.datetime.now()}, cost time: {(time.time()-train_start_time)/3600}h")

    def val(self, epoch):
        self.net.eval()
        losses = AverageMeter()
        # ious = AverageMeter()
        # dices_1s = AverageMeter()
        # dices_2s = AverageMeter()

        with torch.no_grad():
            val_iters = len(self.val_loader)
            for iter_val, (image, mask) in enumerate(self.val_loader):
                image = image.to(self.opt.rank)
                mask = mask.to(self.opt.rank)

                output = self.net(image) 
                loss = self.loss_function(output, mask)
                losses.update(loss.item(), image.shape[0])

                logger.info(f'Val iter: {iter_val+1}/{val_iters}, the loss is {loss.item()}')
                if self.opt.rank == 0:
                    self.writer.add_scalar(f'Loss/val_iter', loss.item(), iter_val+ epoch * len(self.val_loader))

        # log = OrderedDict([
        #     ('loss', losses.avg),
        #     # ('iou', ious.avg),
        #     # ('dice_1', dices_1s.avg),
        #     # ('dice_2', dices_2s.avg)
        # ])
        # return log
                

    def save_checkpoint(self, state_dict, filename='checkpoint.pth'):
        torch.save(state_dict, os.path.join(self.opt.train_output_directory, filename))