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
# from evaluate.metric import iou_score, dice_coef, dice_coef_one
from evaluate import metric
from evaluate.metric import get_metric

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
            # img_paths = glob('/home/jzw/data/LiTS/LITS17/train_image_352*352/*')
            # mask_paths = glob('/home/jzw/data/LiTS/LITS17/train_mask_224*224/*')
            # img_paths = glob('/home/jzw/data/LiTS/LITS17/train_image2d/*')
            img_paths = glob('/home/jzw/data/LiTS/LITS17/train_image_352*352_nospacing/*')
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
        else:
            img_paths = glob('/home/jzw/data/LiTS/LITS17/train_image_352*352_nospacing/*')
            train_img_paths, test_img_paths= \
                train_test_split(img_paths, test_size=0.3, random_state=self.opt.manualSeed)
            test_dataset = LitsDataset(self.opt, test_img_paths)
            self.n_test_img = len(test_dataset)
            test_sampler = DistributedSampler(test_dataset)
            self.test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0, batch_size=1,
                                            pin_memory=True, sampler=test_sampler)
            logger.info('test with {} pair images'.format(self.n_test_img))

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
        elif self.opt.loss.type == 'focal':
            self.loss_function = FocalLoss()
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
            tumor_dscs = AverageMeter()
            tumor_mious = AverageMeter()

            epoch_iters = len(self.train_loader)
            for iter_train, (image, mask) in enumerate(self.train_loader):
                image = image.to(self.opt.rank)
                mask = mask.to(self.opt.rank)

                output = self.net(image)

                # if self.opt.rank == 0:
                #     print(f"image: {image.shape}, {image.min()}, {image.max()}, {image.mean()}")
                #     print(f"mask: {mask.shape}, {mask.min()}, {mask.max()}, {mask.mean()}")
                #     print(f"output: {output.shape}, {output.min()}, {output.max()}, {output.mean()}")

                loss = self.loss_function(output, mask)

                liver_dsc, liver_miou, liver_acc, liver_ppv, liver_sen, liver_hd = get_metric(output.detach()[:, 0, :, :], mask.detach()[:, 0, :, :], self.opt.thr)
                tumor_dsc, tumor_miou, tumor_acc, tumor_ppv, tumor_sen, tumor_hd = get_metric(output.detach()[:, 1, :, :], mask.detach()[:, 1, :, :], self.opt.thr)

                losses.update(loss.item(), image.shape[0])

                liver_dscs.update(liver_dsc, image.shape[0])
                liver_mious.update(liver_miou, image.shape[0])
                tumor_dscs.update(tumor_dsc, image.shape[0])
                tumor_mious.update(tumor_miou, image.shape[0])

                # compute gradient and do optimizing step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logger.info(f'Training Epoch: {epoch+1}/{self.opt.train_epoch},iter: {iter_train+1}/{epoch_iters}, the loss is {loss.item()}, liver_dsc:{liver_dsc}, tumor_dsc:{tumor_dsc}, out_min:{output.min()}, out_max:{output.max()}, out_mean:{output.mean()}')
               
            self.lr_scheduler.step()

            if self.opt.rank == 0:
                self.writer.add_scalar(f'loss/train_epoch', losses.avg, epoch) # 平均每条数据的loss，即batch=1
                self.writer.add_scalar(f'liver_dscs/train_epoch', liver_dscs.avg, epoch)
                self.writer.add_scalar(f'liver_mious/train_epoch', liver_mious.avg, epoch)
                self.writer.add_scalar(f'tumor_dscs/train_epoch', tumor_dscs.avg, epoch)
                self.writer.add_scalar(f'tumor_mious/train_epoch', tumor_mious.avg, epoch)

            logger.info(f'Start evalute at Epoch: {epoch+1}/{self.opt.train_epoch}')
            val_res = self.val(epoch)

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

        logger.info(f"Finish training at {datetime.datetime.now()}, cost time: {(time.time()-train_start_time)/3600}h")

    def val(self, epoch):
        self.net.eval()
        losses = AverageMeter()
        liver_dscs = AverageMeter()
        liver_mious = AverageMeter()
        tumor_dscs = AverageMeter()
        tumor_mious = AverageMeter()

        with torch.no_grad():
            val_iters = len(self.val_loader)
            for iter_val, (image, mask) in enumerate(self.val_loader):
                image = image.to(self.opt.rank)
                mask = mask.to(self.opt.rank)

                output = self.net(image) 

                loss = self.loss_function(output, mask)

                liver_dsc, liver_miou, liver_acc, liver_ppv, liver_sen, liver_hd = get_metric(torch.sigmoid(output.detach()[:, 0, :, :]), mask.detach()[:, 0, :, :], self.opt.thr)
                tumor_dsc, tumor_miou, tumor_acc, tumor_ppv, tumor_sen, tumor_hd = get_metric(torch.sigmoid(output.detach()[:, 1, :, :]), mask.detach()[:, 1, :, :], self.opt.thr)

                losses.update(loss.item(), image.shape[0])

                liver_dscs.update(liver_dsc, image.shape[0])
                liver_mious.update(liver_miou, image.shape[0])
                tumor_dscs.update(tumor_dsc, image.shape[0])
                tumor_mious.update(tumor_miou, image.shape[0])

                logger.info(f'Val iter: {iter_val+1}/{val_iters}, the loss is {loss.item()}')

            if self.opt.rank == 0:
                self.writer.add_scalar(f'loss/val_epoch', losses.avg, epoch)
                self.writer.add_scalar(f'liver_dscs/val_epoch', liver_dscs.avg, epoch)
                self.writer.add_scalar(f'liver_mious/val_epoch', liver_mious.avg, epoch)
                self.writer.add_scalar(f'tumor_dscs/val_epoch', tumor_dscs.avg, epoch)
                self.writer.add_scalar(f'tumor_mious/val_epoch', tumor_mious.avg, epoch)

        self.net.train()

        val_res = {
            'loss': losses.avg,
            'liver_dscs': liver_dscs.avg,
            'liver_mious': liver_mious.avg,
            'tumor_dscs': tumor_dscs.avg,
            'tumor_mious': tumor_mious.avg
        }
        return val_res
                
    def test(self):
        self.net.eval()
        # checkpoint_root = Path("runs/lits_seg/unet/train_2021-09-16-22-20-37/checkpoints")
        # model_name = "epoch_100_model_best_loss.pth"
        # checkpint_path = checkpint_root / model_name

        # ious = AverageMeter()
        ious_1s = AverageMeter()
        ious_2s = AverageMeter()
        mious_1s = AverageMeter()
        mious_2s = AverageMeter()
        dices_1s = AverageMeter()
        dices_2s = AverageMeter()

        # self.load_checkpoint(self.net, checkpint_path)

        with torch.no_grad():
            test_iters = len(self.test_loader)
            for iter_test, (image, mask) in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                image = image.to(self.opt.rank)
                mask = mask.to(self.opt.rank)
                output = self.net(image) 
                assert (image.shape[0] == 1 and mask.shape[0] == 1), "Please set batch size = 1 when test."
                # print(output.shape)
                # print(mask.shape)

                # print(output[0][1].min(), output[0][1].max(), output[0][1].mean())
                # print(mask[0][1].min(), mask[0][1].max(), mask[0][1].mean())
                miou_1 = metric.mean_iou(output[0][0], mask[0][0]) 
                miou_2 = metric.mean_iou(output[0][1], mask[0][1]) 
                iou_1 = metric.iou_score(output[0][0], mask[0][0]) 
                iou_2 = metric.iou_score(output[0][1], mask[0][1]) 
                # print(iou_1)
                # print(iou_2)
                # iou_1 = metric.iou_score(output[0][0], mask[0][0]) 
                # iou_2 = metric.iou_score(output[0][1], mask[0][1]) 
                # print(iou_1)
                # print(iou_2)
                # assert 1>4
                dice_1 = metric.dice_coef(output, mask)[0]
                dice_2 = metric.dice_coef(output, mask)[1]

                mious_1s.update(miou_1, image.shape[0])
                mious_2s.update(miou_2, image.shape[0])
                ious_1s.update(iou_1, image.shape[0])
                ious_2s.update(iou_2, image.shape[0]) 
                dices_1s.update(dice_1, image.shape[0])
                dices_2s.update(dice_2, image.shape[0])

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
        # print('*' * 15, self.model_name + ':', '*' * 15)

        logger.info(f"{'*' * 15} liver_mIoU: {mious_1s.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} liver_IouScore: {ious_1s.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} liver_DiceScore: {dices_1s.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} tumor_mIoU: {mious_2s.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} tumor_IouScore: {ious_2s.avg} {'*' * 15}")
        logger.info(f"{'*' * 15} tumor_DiceScore: {dices_2s.avg} {'*' * 15}")

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