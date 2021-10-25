import os
from glob import glob
from pathlib import Path
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
# from loss.loss import FocalLoss
from loss.focal_loss import FocalLoss
from loss.dice_loss import DiceLoss
# from evaluate.metric import iou_score, dice_coef, dice_coef_one
from evaluate import metric
from evaluate.metric import get_metric

from dataset.lits.lits_2d import LitsDataset, LitsLiverDataset, LitsTumorDataset
from dataset.lits.LitsMyself import Lits_DataSet

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
        if self.opt.net_name == "TransUNet":
            vit_name = "R50-ViT-B_16"
            num_classes = 2
            img_size = 512
            vit_patches_size = 16
            n_skip = 3
            config_vit = build_net(self.opt.net_name)[1][vit_name]
            config_vit.n_classes = num_classes
            config_vit.n_skip = n_skip
            if vit_name.find('R50') != -1:
                config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

            self.net = build_net(self.opt.net_name)[0](config_vit, img_size=img_size, num_classes=config_vit.n_classes).to(self.opt.rank)
        else:
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
            # ## 加载 lits_2d.LitsDataset(LitsLiverDataset, LitsTumorDataset)
            # img_paths = glob('/home/jzw/data/LiTS/LITS17/train_image_352*352_no-9/*')
            # if self.opt.debug:
            #     img_paths = img_paths[:80]
            # train_img_paths, val_img_paths= \
            #     train_test_split(img_paths, test_size=0.3, random_state=self.opt.manualSeed)
            # train_dataset = LitsDataset(self.opt, train_img_paths)
            # train_dataset = LitsLiverDataset(self.opt, train_img_paths)
            # train_dataset = LitsTumorDataset(self.opt, train_img_paths)
            ## 加载 LitsMyself.Lits_DataSet
            if self.opt.debug:
                train_dataset = Lits_DataSet(self.opt.frame_num, 1, self.opt.train_dir, mode='debug') 
            else:
                train_dataset = Lits_DataSet(self.opt.frame_num, 1, self.opt.train_dir, mode='train') 

            self.n_train_img = len(train_dataset)
            self.max_iter = self.n_train_img * self.opt.train_epoch // self.opt.batch_size // self.opt.world_size
            train_sampler = DistributedSampler(train_dataset)
            self.train_loader = DataLoader(train_dataset, shuffle=False, num_workers=0, batch_size=self.opt.batch_size,
                                           pin_memory=True, sampler=train_sampler)
            logger.info('train with {} pair images'.format(self.n_train_img))

            # ## 加载 lits_2d.LitsDataset(LitsLiverDataset, LitsTumorDataset)
            # val_dataset = LitsDataset(self.opt, val_img_paths)
            # val_dataset = LitsLiverDataset(self.opt, val_img_paths)
            # val_dataset = LitsTumorDataset(self.opt, val_img_paths)
            ## 加载 LitsMyself.Lits_DataSet
            if self.opt.debug:
                val_dataset = Lits_DataSet(self.opt.frame_num, 1, self.opt.train_dir, mode='debug') 
            else:
                val_dataset = Lits_DataSet(self.opt.frame_num, 1, self.opt.train_dir, mode='test') 

            self.n_val_img = len(val_dataset)
            val_sampler = DistributedSampler(val_dataset)
            self.val_loader = DataLoader(val_dataset, shuffle=False, num_workers=0, batch_size=1,
                                            pin_memory=True, sampler=val_sampler)
            logger.info('val with {} pair images'.format(self.n_val_img))
        else:
            # ## 加载 lits_2d.LitsDataset(LitsLiverDataset, LitsTumorDataset)
            # img_paths = glob('/home/jzw/data/LiTS/LITS17/train_image_352*352_no-9/*')
            # train_img_paths, test_img_paths= \
            #     train_test_split(img_paths, test_size=0.3, random_state=self.opt.manualSeed)
            # test_dataset = LitsDataset(self.opt, test_img_paths)
            # test_dataset = LitsLiverDataset(self.opt, test_img_paths)
            # test_dataset = LitsTumorDataset(self.opt, test_img_paths)
            ## 加载 LitsMyself.Lits_DataSet
            if self.opt.debug:
                test_dataset = Lits_DataSet(self.opt.frame_num, 1, self.opt.train_dir, mode='debug') 
            else:
                test_dataset = Lits_DataSet(self.opt.frame_num, 1, self.opt.train_dir, mode='test')             
            self.n_test_img = len(test_dataset)
            test_sampler = DistributedSampler(test_dataset)
            self.test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0, batch_size=1,
                                            pin_memory=True, sampler=test_sampler)

            logger.info('test with {} pair images'.format(self.n_test_img))

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
        elif self.opt.loss.type == 'focal':
            self.loss_function = FocalLoss()
        elif self.opt.loss.type == 'dice':
            self.loss_function = DiceLoss()
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
            self.writer = SummaryWriter(self.opt.summary_directory)

        train_start_time = time.time()
        best_loss = 1e10
        best_iou = 0
        saved_trigger = 0
        saved_epoch = set()

        for epoch in range(self.opt.train_epoch):
            losses = AverageMeter()
            liver_dscs = AverageMeter()
            liver_mious = AverageMeter()
            liver_accs = AverageMeter()
            liver_ppvs = AverageMeter()
            liver_sens = AverageMeter()
            liver_hds = AverageMeter()
            tumor_dscs = AverageMeter()
            tumor_mious = AverageMeter()
            tumor_accs = AverageMeter()
            tumor_ppvs = AverageMeter()
            tumor_sens = AverageMeter()
            tumor_hds = AverageMeter()

            epoch_iters = len(self.train_loader)
            for iter_train, (image, mask) in enumerate(self.train_loader):
                image = image.to(self.opt.rank)
                mask = mask.to(self.opt.rank)

                output = self.net(image)

                # if self.opt.rank == 0:
                #     # print(f"image: {image.shape}, {image.min()}, {image.max()}, {image.mean()}, {image.std()}")
                #     # print(f"mask: {mask.shape}, {mask.min()}, {mask.max()}, {mask.mean()}, {image.std()}")
                #     logger.info(f"output: {output.shape}, {output.min()}, {output.max()}, {output.mean()}, {output.std()}")

                loss = self.loss_function(output, mask)

                liver_dsc, liver_miou, liver_acc, liver_ppv, liver_sen, liver_hd = get_metric(output.detach()[:, 0, :, :], mask.detach()[:, 0, :, :], self.opt.thr)
                tumor_dsc, tumor_miou, tumor_acc, tumor_ppv, tumor_sen, tumor_hd = get_metric(output.detach()[:, 1, :, :], mask.detach()[:, 1, :, :], self.opt.thr)
                # tumor_dsc, tumor_miou, tumor_acc, tumor_ppv, tumor_sen, tumor_hd = get_metric(torch.sigmoid(output.detach()[:, 0, :, :]), mask.detach()[:, 0, :, :], self.opt.thr)

                losses.update(loss.item(), image.shape[0])
                
                liver_dscs.update(liver_dsc, image.shape[0])
                liver_accs.update(liver_acc, image.shape[0])
                liver_ppvs.update(liver_ppv, image.shape[0])
                liver_sens.update(liver_sen, image.shape[0])
                # liver_mious.update(liver_miou, image.shape[0])
                # liver_hds.update(liver_hd, image.shape[0])
                tumor_dscs.update(tumor_dsc, image.shape[0])
                tumor_accs.update(tumor_acc, image.shape[0])
                tumor_ppvs.update(tumor_ppv, image.shape[0])
                tumor_sens.update(tumor_sen, image.shape[0])
                # tumor_mious.update(tumor_miou, image.shape[0])
                # tumor_hds.update(tumor_hd, image.shape[0])

                ## compute gradient and do optimizing step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # logger.info(f'Training Epoch: {epoch+1}/{self.opt.train_epoch},iter: {iter_train+1}/{epoch_iters}, the loss is {loss.item()}, liver_dsc:{liver_dsc}, tumor_dsc:{tumor_dsc}, liver_min:{output.detach()[:,0,:,:].min()}, liver_max:{output.detach()[:,0,:,:].max()}, tumor_min:{output.detach()[:,1,:,:].min()}, tumor_max:{output.detach()[:,1,:,:].max()}')
                # logger.info(f'Training Epoch: {epoch+1}/{self.opt.train_epoch},iter: {iter_train+1}/{epoch_iters}, the loss is {loss.item()}, liver_dsc:{liver_dsc}, liver_sen:{liver_sen}')
                logger.info(f'Training Epoch: {epoch+1}/{self.opt.train_epoch},iter: {iter_train+1}/{epoch_iters}, loss: {loss.item()}, liver_dsc:{liver_dsc}, liver_ppv:{liver_ppv}, liver_sen:{liver_sen}, tumor_dsc:{tumor_dsc}, tumor_ppv:{tumor_ppv}, tumor_sen:{tumor_sen}, output:[{output.min()},{output.max()},{output.mean()}]')

                if self.opt.rank == 0:
                    self.writer.add_scalar(f'loss/train_iter', loss.item(), epoch*epoch_iters+iter_train)
                    self.writer.add_scalar(f'liver_dscs/train_iter', liver_dsc, epoch*epoch_iters+iter_train)
                    self.writer.add_scalar(f'liver_accs/train_iter', liver_acc, epoch*epoch_iters+iter_train)
                    self.writer.add_scalar(f'liver_ppvs/train_iter', liver_ppv, epoch*epoch_iters+iter_train)
                    self.writer.add_scalar(f'liver_sens/train_iter', liver_sen, epoch*epoch_iters+iter_train)
                    self.writer.add_scalar(f'tumor_dscs/train_iter', tumor_dsc, epoch*epoch_iters+iter_train)
                    self.writer.add_scalar(f'tumor_accs/train_iter', tumor_acc, epoch*epoch_iters+iter_train)
                    self.writer.add_scalar(f'tumor_ppvs/train_iter', tumor_ppv, epoch*epoch_iters+iter_train)
                    self.writer.add_scalar(f'tumor_sens/train_iter', tumor_sen, epoch*epoch_iters+iter_train)
               
            self.lr_scheduler.step()

            logger.info(f'Finish Training Epoch: {epoch+1}/{self.opt.train_epoch}, loss: {losses.avg}, liver_dsc:{liver_dscs.avg}, liver_acc:{liver_accs.avg}, liver_ppv:{liver_ppvs.avg}, liver_sen:{liver_sens.avg}, tumor_dsc:{tumor_dscs.avg}, tumor_acc:{tumor_accs.avg}, tumor_ppv:{tumor_ppvs.avg}, tumor_sen:{tumor_sens.avg}')
            if self.opt.rank == 0:
                self.writer.add_scalar(f'loss/train_epoch', losses.avg, epoch) 
                self.writer.add_scalar(f'liver_dscs/train_epoch', liver_dscs.avg, epoch)
                self.writer.add_scalar(f'liver_accs/train_epoch', liver_accs.avg, epoch)
                self.writer.add_scalar(f'liver_ppvs/train_epoch', liver_ppvs.avg, epoch)
                self.writer.add_scalar(f'liver_sens/train_epoch', liver_sens.avg, epoch)
                self.writer.add_scalar(f'tumor_dscs/train_epoch', tumor_dscs.avg, epoch)
                self.writer.add_scalar(f'tumor_accs/train_epoch', tumor_accs.avg, epoch)
                self.writer.add_scalar(f'tumor_ppvs/train_epoch', tumor_ppvs.avg, epoch)
                self.writer.add_scalar(f'tumor_sens/train_epoch', tumor_sens.avg, epoch)

            logger.info(f'Start evalute at Epoch: {epoch+1}/{self.opt.train_epoch}')
            # val_res = self.val(epoch)
            self.val(epoch)

            saved_trigger += 1 
            if self.opt.rank == 0:
                # if epoch >= int(self.opt.train_epoch // 2) and epoch not in saved_epoch and val_res["loss"] < best_loss:
                #     saved_epoch.add(epoch)
                #     saved_trigger = 0
                #     self.save_checkpoint({'epoch': self.opt.train_epoch,
                #                         'arch': self.opt.net_name,
                #                         'state_dict': self.net.state_dict(),
                #                         }, f'epoch_{epoch+1}_model_best_loss.pth')
                # if epoch >= int(self.opt.train_epoch // 2) and epoch not in saved_epoch and val_res["iou"] > best_iou:
                #     saved_epoch.add(epoch)
                #     saved_trigger = 0
                #     self.save_checkpoint({'epoch': self.opt.train_epoch,
                #                         'arch': self.opt.net_name,
                #                         'state_dict': self.net.state_dict(),
                #                         }, f'epoch_{epoch+1}_model_best_iou.pth')
                if epoch not in saved_epoch and (epoch+1) % self.opt.save_every_epoch == 0:
                    saved_epoch.add(epoch)
                    self.save_checkpoint({'epoch': self.opt.train_epoch,
                                        'arch': self.opt.net_name,
                                        'state_dict': self.net.state_dict(),
                                        }, f'epoch_{epoch+1}_model.pth')
                
                # if not self.opt.early_stop is None:
                #     if saved_trigger >= self.opt.early_stop:
                #         logger.info(f'Early stop at Epoch: {epoch+1}/{self.opt.train_epoch}')
                #         break
            logger.info(f'Finish Epoch: {epoch+1}/{self.opt.train_epoch}')
        logger.info(f"Finish training at {datetime.datetime.now()}, cost time: {(time.time()-train_start_time)/3600}h")

    def val(self, epoch):
        self.net.eval()
        losses = AverageMeter()
        liver_dscs = AverageMeter()
        liver_mious = AverageMeter()
        liver_accs = AverageMeter()
        liver_ppvs = AverageMeter()
        liver_sens = AverageMeter()
        liver_hds = AverageMeter()
        tumor_dscs = AverageMeter()
        tumor_mious = AverageMeter()
        tumor_accs = AverageMeter()
        tumor_ppvs = AverageMeter()
        tumor_sens = AverageMeter()
        tumor_hds = AverageMeter()

        with torch.no_grad():
            val_iters = len(self.val_loader)
            for iter_val, (image, mask) in enumerate(self.val_loader):
                image = image.to(self.opt.rank)
                mask = mask.to(self.opt.rank)

                output = self.net(image) 

                loss = self.loss_function(output, mask)

                liver_dsc, liver_miou, liver_acc, liver_ppv, liver_sen, liver_hd = get_metric(output.detach()[:, 0, :, :], mask.detach()[:, 0, :, :], self.opt.thr)
                tumor_dsc, tumor_miou, tumor_acc, tumor_ppv, tumor_sen, tumor_hd = get_metric(output.detach()[:, 1, :, :], mask.detach()[:, 1, :, :], self.opt.thr)
                # tumor_dsc, tumor_miou, tumor_acc, tumor_ppv, tumor_sen, tumor_hd = get_metric(torch.sigmoid(output.detach()[:, 0, :, :]), mask.detach()[:, 0, :, :], self.opt.thr)

                losses.update(loss.item(), image.shape[0])

                liver_dscs.update(liver_dsc, image.shape[0])
                liver_accs.update(liver_acc, image.shape[0])
                liver_ppvs.update(liver_ppv, image.shape[0])
                liver_sens.update(liver_sen, image.shape[0])
                liver_hds.update(liver_hd, image.shape[0])
                tumor_dscs.update(tumor_dsc, image.shape[0])
                tumor_accs.update(tumor_acc, image.shape[0])
                tumor_ppvs.update(tumor_ppv, image.shape[0])
                tumor_sens.update(tumor_sen, image.shape[0])
                tumor_hds.update(tumor_hd, image.shape[0])

                logger.info(f'Val iter: {iter_val+1}/{val_iters}, liver_dsc:{liver_dsc}, liver_acc:{liver_acc}, liver_ppv:{liver_ppv}, liver_sen:{liver_sen}, tumor_dsc:{tumor_dsc}, tumor_acc:{tumor_acc}, tumor_:ppv:{tumor_ppv}, tumor_sen:{tumor_sen}, output:[{output.min()},{output.max()},{output.mean()}]')
                if self.opt.rank == 0:
                    self.writer.add_scalar(f'loss/val_iter', loss.item(), epoch*val_iters+iter_val)
                    self.writer.add_scalar(f'liver_dscs/val_iter', liver_dsc, epoch*val_iters+iter_val)
                    self.writer.add_scalar(f'liver_accs/val_iter', liver_acc, epoch*val_iters+iter_val)
                    self.writer.add_scalar(f'liver_ppvs/val_iter', liver_ppv, epoch*val_iters+iter_val)
                    self.writer.add_scalar(f'liver_sens/val_iter', liver_sen, epoch*val_iters+iter_val)
                    self.writer.add_scalar(f'tumor_dscs/val_iter', tumor_dsc, epoch*val_iters+iter_val)
                    self.writer.add_scalar(f'tumor_accs/val_iter', tumor_acc, epoch*val_iters+iter_val)
                    self.writer.add_scalar(f'tumor_ppvs/val_iter', tumor_ppv, epoch*val_iters+iter_val)
                    self.writer.add_scalar(f'tumor_sens/val_iter', tumor_sen, epoch*val_iters+iter_val)

            logger.info(f'Finish Val Epoch: {epoch+1}/{self.opt.train_epoch}, loss: {losses.avg}, liver_dsc:{liver_dscs.avg}, liver_acc:{liver_accs.avg}, liver_ppv:{liver_ppvs.avg}, liver_sen:{liver_sens.avg}, tumor_dsc:{tumor_dscs.avg}, tumor_acc:{tumor_accs.avg}, tumor_ppv:{tumor_ppvs.avg}, tumor_sen:{tumor_sens.avg}')
            if self.opt.rank == 0:
                self.writer.add_scalar(f'loss/val_epoch', losses.avg, epoch) # 平均每条数据的loss，即batch=1
                self.writer.add_scalar(f'liver_dscs/val_epoch', liver_dscs.avg, epoch)
                self.writer.add_scalar(f'liver_accs/val_epoch', liver_accs.avg, epoch)
                self.writer.add_scalar(f'liver_ppvs/val_epoch', liver_ppvs.avg, epoch)
                self.writer.add_scalar(f'liver_sens/val_epoch', liver_sens.avg, epoch)
                self.writer.add_scalar(f'liver_hds/val_epoch', liver_hds.avg, epoch)
                self.writer.add_scalar(f'tumor_dscs/val_epoch', tumor_dscs.avg, epoch)
                self.writer.add_scalar(f'tumor_accs/val_epoch', tumor_accs.avg, epoch)
                self.writer.add_scalar(f'tumor_ppvs/val_epoch', tumor_ppvs.avg, epoch)
                self.writer.add_scalar(f'tumor_sens/val_epoch', tumor_sens.avg, epoch)
                self.writer.add_scalar(f'tumor_hds/val_epoch', tumor_hds.avg, epoch)

        self.net.train()

        # val_res = {
        #     'loss': losses.avg,
        #     'liver_dscs': liver_dscs.avg,
        #     'liver_mious': liver_mious.avg,
        #     # 'tumor_dscs': tumor_dscs.avg,
        #     # 'tumor_mious': tumor_mious.avg
        # }
        # return val_res
                
    def test(self):
        self.net.eval()
        if self.opt.rank == 0:
            self.writer = SummaryWriter(self.opt.summary_directory)

        losses = AverageMeter()
        liver_dscs = AverageMeter()
        liver_accs = AverageMeter()
        liver_ppvs = AverageMeter()
        liver_sens = AverageMeter()
        tumor_dscs = AverageMeter()
        tumor_accs = AverageMeter()
        tumor_ppvs = AverageMeter()
        tumor_sens = AverageMeter()

        # self.load_checkpoint(self.net, checkpint_path)

        with torch.no_grad():
            test_iters = len(self.test_loader)
            saved_id = 0
            for iter_test, (image, mask) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                image = image.to(self.opt.rank)
                mask = mask.to(self.opt.rank)
                output = self.net(image) 

                loss = self.loss_function(output, mask)
                assert (image.shape[0] == 1 and mask.shape[0] == 1), "Please set batch size = 1 when test."

                liver_dsc, liver_miou, liver_acc, liver_ppv, liver_sen, liver_hd = get_metric(torch.sigmoid(output.detach()[:, 0, :, :, :]), mask.detach()[:, 0, :, :, :], self.opt.thr)
                tumor_dsc, tumor_miou, tumor_acc, tumor_ppv, tumor_sen, tumor_hd = get_metric(torch.sigmoid(output.detach()[:, 1, :, :, :]), mask.detach()[:, 1, :, :, :], self.opt.thr)

                logger.info(f'iter: {iter_test+1}/{len(self.test_loader)}, loss: {loss.item()}, liver_dsc:{liver_dsc}, liver_acc:{liver_acc}, liver_ppv:{liver_ppv}, liver_sen:{liver_sen}, tumor_dsc:{tumor_dsc}, tumor_acc:{tumor_acc}, tumor_:ppv:{tumor_ppv}, tumor_sen:{tumor_sen}, output:[{output.min()},{output.max()},{output.mean()}]')            

                losses.update(loss, image.shape[0])
                liver_dscs.update(liver_dsc,  image.shape[0])
                liver_accs.update(liver_acc,  image.shape[0])
                liver_ppvs.update(liver_ppv, image.shape[0])
                liver_sens.update(liver_sen, image.shape[0])
                tumor_dscs.update(tumor_dsc, image.shape[0])
                tumor_accs.update(tumor_acc, image.shape[0])
                tumor_ppvs.update(tumor_ppv, image.shape[0])
                tumor_sens.update(tumor_sen, image.shape[0])

                if self.opt.rank == 0:
                    self.writer.add_scalar(f'loss/iter_test', loss.item(), iter_test)
                    self.writer.add_scalar(f'liver_dscs/iter_test', liver_dsc, iter_test)
                    self.writer.add_scalar(f'liver_accs/iter_test', liver_acc, iter_test)
                    self.writer.add_scalar(f'liver_ppvs/iter_test', liver_ppv, iter_test)
                    self.writer.add_scalar(f'liver_sens/iter_test', liver_sen, iter_test)
                    self.writer.add_scalar(f'tumor_dscs/iter_test', tumor_dsc, iter_test)
                    self.writer.add_scalar(f'tumor_accs/iter_test', tumor_acc, iter_test)
                    self.writer.add_scalar(f'tumor_ppvs/iter_test', tumor_ppv, iter_test)
                    self.writer.add_scalar(f'tumor_sens/iter_test', tumor_sen, iter_test)

                ## save output images
                # logger.info("Save visual images.")
                # if saved_id < 100 and mask[0][0].max() > 1e-5:
                #     # print(mask.shape)
                #     for depth in range(mask.shape[2]):
                #         if mask[0, 1, depth:depth+1, :, :].max() > 1e-5:
                #             saved_id += 1
                #             saved_mask0 = (mask[0, 0, depth:depth+1, :, :].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
                #             cv2.imwrite(str(Path(self.opt.visual_directory) / f"{str(iter_test)}_{depth}_liver_mask.png"), saved_mask0)
                #             saved_output0 = (output[0, 0, depth:depth+1, :, :].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8) 
                #             cv2.imwrite(str(Path(self.opt.visual_directory) / f"{str(iter_test)}_{depth}_liver_out.png"), saved_output0)
                #             saved_mask1 = (mask[0, 1, depth:depth+1, :, :].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
                #             cv2.imwrite(str(Path(self.opt.visual_directory) / f"{str(iter_test)}_{depth}_tumor_mask.png"), saved_mask1)
                #             saved_output1 = (output[0, 1, depth:depth+1, :, :].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8) 
                #             cv2.imwrite(str(Path(self.opt.visual_directory) / f"{str(iter_test)}_{depth}_tumor_out.png"), saved_output1)


                # logger.info("Save visual images.")
                # print("Save visual images.")
                if True:#iter_test < 500:
                    # print(mask.shape)
                    if True:#mask[0][1:2].sum() > 0:
                        logger.info(f"{output[0][:1].min()}, {output[0][:1].max()}, {output[0][1:2].min()}, {output[0][1:2].max()}")
                        # print(mask[0][:1].sum())
                        # print(iter_test)
                    # print(type(mask[0][:1]), mask[0][:1].shape)
                    # assert 1>4
                    # cv2.imwrite(str(Path(self.opt.visual_directory) / f"{str(iter_test)}_img.png"), image[0][0])
                    # cv2.imwrite(str(Path(self.opt.visual_directory) / f"{str(iter_test)}_liver_out.png"), output[0][0])
                        saved_mask0 = (mask[0][:1].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
                        # cv2.imwrite(str(Path(self.opt.visual_directory) / f"{str(iter_test)}_liver_mask.png"), saved_mask0)
                        saved_output0 = (output[0][:1].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8) 
                        # cv2.imwrite(str(Path(self.opt.visual_directory) / f"{str(iter_test)}_liver_out.png"), saved_output0)
                        saved_mask1 = (mask[0][1:2].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
                        # cv2.imwrite(str(Path(self.opt.visual_directory) / f"{str(iter_test)}_tumor_mask.png"), saved_mask1)
                        saved_output1 = (output[0][1:2].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8) 
                        # uni = np.unique(saved_output1)
                        # if set(uni) != {0}:
                        #     logger.info(uni)
                        # cv2.imwrite(str(Path(self.opt.visual_directory) / f"{str(iter_test)}_tumor_out.png"), saved_output1)
                        # assert 1>4

                # logger.info(f'Val iter: {iter_test+1}/{val_iters}, the loss is {loss.item()}')

        logger.info(f"{'*' * 15} loss {losses.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} liver_dsc {liver_dscs.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} liver_acc: {liver_accs.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} liver_ppv: {liver_ppvs.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} liver_sen: {liver_sens.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} tumor_dsc: {tumor_dscs.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} tumor_acc: {tumor_accs.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} tumor_ppv: {tumor_ppvs.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} tumor_sen: {tumor_sens.avg} {'*' * 15}")

    def save_checkpoint(self, state_dict, filename='checkpoint.pth'):
        torch.save(state_dict, os.path.join(self.opt.checkpoint_directory, filename))

    def load_checkpoint(self, model, checkpoint_path):
        loaded_dict = torch.load(checkpoint_path)
        state_dict = loaded_dict[state_dict]
        model.load_state_dict(state_dict)
    
    def load_state_keywise(self, model_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.opt.rank}
        resume_dict = torch.load(model_path, map_location=map_location)
        if 'state_dict' in resume_dict.keys():
            resume_dict = resume_dict['state_dict']
        self.net.load_state_dict(resume_dict)