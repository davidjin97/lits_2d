import sys
import os
from glob import glob
import numpy as np
from collections import  Counter
from pathlib import Path
import cv2
# from utils.common import *
# from scipy import ndimage
import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
# np.random.seed(0)

class LitsDataset(Dataset):
    def __init__(self, args, img_paths):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = list(map(lambda x: x.replace('image', 'mask').replace('slice','seg'), self.img_paths))
        # self.img_root = img_root
        # self.mask_root = mask_root
        # self.img_paths, self.mask_paths = self.get_data_paths()
        # print(self.img_paths[:5])
        # self.img_paths = img_paths
        # self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = str(self.img_paths[idx])
        mask_path = str(self.mask_paths[idx])
        # seg_volume_num = seg_name.split("_")[0]
        # print(img_path)
        # print(mask_path)
        npimage = np.load(img_path) # (1, 224, 224)
        npmask = np.load(mask_path) # (1, 224, 224) 0表示背景，1表示肝脏，2表示肝脏肿瘤
        # print(npimage.shape)
        # print(npmask.shape)
        # npimage = npimage.transpose((2, 0, 1))[:1, :, :]
        # npmask = npmask[np.newaxis, :, :]
        # print(f"npimage0: {npimage.shape}, {npimage.min()}, {npimage.max()}, {npimage.mean()}")
        # print(f"npmask: {npmask.shape}, {npmask.min()}, {npmask.max()}, {npmask.mean()}")
        # assert 1>8

        # npimage = npimage[0, :, :, np.newaxis] # (224, 224, 1)
        # npimage = npimage.transpose((2, 0, 1)) # (1, 224, 224)

        npmask = npmask[0, :, :]

        # 拆分 liver 和 tumor label
        liver_label = npmask.copy()
        liver_label[npmask == 2] = 1
        liver_label[npmask == 1] = 1

        tumor_label = npmask.copy()
        tumor_label[npmask == 1] = 0
        tumor_label[npmask == 2] = 1

        _, h, w = npimage.shape
        nplabel = np.empty((2, w, h))

        nplabel[0, :, :] = liver_label
        nplabel[1, :, :] = tumor_label

        # nplabel = nplabel.transpose((2, 0, 1))

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")
        # print(npimage.shape ,nplabel.shape)

        return npimage, nplabel#, img_path # (1, 224, 224), (2, 224, 224)
        # return npimage, nplabel[1:, :, :] # 只有tumor的mask(1, 224, 224), (1, 224, 224)
    
    def get_data_paths(self):
        img_paths = list(self.img_root.iterdir())
        # print(img_paths[:4])
        # print(type(img_paths[1]))
        # print(type(img_paths))
        # img_paths = glob(str(img_root))
        # print(str(img_paths[0]).replace('image', 'mask'))

        mask_paths = list(map(lambda x: Path(str(x).replace('image', 'mask').replace('slice', 'seg')), img_paths))# .replace('slice', 'seg')
        # print(mask_paths[0].exists())
        # assert 1>4
        return img_paths, mask_paths 

'''
class LitsDataSet(Dataset):
    def __init__(self, raw_data_path, mode=None):
        self.raw_data_path = raw_data_path

        # if mode=='train':
        #     self.filename_list = load_file_name_list(os.path.join(dataset_path, 'train_name_list.txt'))
        # elif mode =='val':
        #     self.filename_list = load_file_name_list(os.path.join(dataset_path, 'val_name_list.txt'))
        # else:
        #     raise TypeError('Dataset mode error!!! ')

        self.data = self.get_train_data(self.frameNum, 1)
        # self.n_labels = 3

    def __getitem__(self, index):
        img, target = self.data[index][0], self.data[index][1]
        img = img.astype(np.float32)
        target = target.astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(target)

    def __len__(self):
        return len(self.data)

    def clamp(self, niiImage):
        ans = niiImage.copy()
        ans[ans < -200] = -200
        ans[ans > 250] = 250
        maxx = np.max(ans)
        minn = np.min(ans)
        # ansMean = ans.mean()
        # ansStd =  ans.std()
        maxx = maxx + 1 if maxx == minn else maxx # 如果minn和maxx相等，为了不除0，将maxx加1
        return 2 * (ans - minn) / (maxx - minn) - 1 #[-1,1]

    def fold(self, niiImage, mask, flag):
        sh = niiImage.shape
        tmpNii = np.empty((sh[0], sh[1], sh[2]))
        tmpMask = np.empty((sh[0], sh[1], sh[2]))
        if flag == 1:#上下翻折
            for xi in range(sh[1]):
                for yi in range(sh[2]):
                    tmpNii[:, xi, yi] = niiImage[:, sh[1] - xi - 1, yi]
                    tmpMask[:, xi, yi] = mask[:, sh[1] - xi - 1, yi]
            return tmpNii, tmpMask
        if flag == 2:#左右翻折
            for xi in range(sh[1]):
                for yi in range(sh[2]):
                    tmpNii[:, xi, yi] = niiImage[:, xi, sh[2] - yi - 1]
                    tmpMask[:, xi, yi] = mask[:, xi, sh[2] - yi - 1]
            return tmpNii, tmpMask
        if flag == 3:#中心旋转
            for xi in range(sh[1]):
                for yi in range(sh[2]):
                    tmpNii[:, xi, yi] = niiImage[:, sh[1] - xi - 1, sh[2] - yi - 1]
                    tmpMask[:, xi, yi] = mask[:, sh[1] - xi - 1, sh[2] - yi - 1]
            return tmpNii, tmpMask

    def randomCrop(self, niiImage, mask, minx, maxy):
        """
        实现上下方向裁剪，minx，maxy分别是裁剪起始点x坐标的取值的上下限
        16x512x512----->16x352x512
        
        """
        # tx = np.random.randint(minx, maxy + 1)
        tx = 80
        tmpNii = np.zeros((16, 352, 512))
        tmpNii[:, :, :] = niiImage[:, tx:tx + 352, :]
        tmpMask = np.zeros((16, 352, 512))
        tmpMask[:, :, :] = mask[:, tx:tx + 352, :]
        return tmpNii, tmpMask

    def seperateLiverTumorMask(self, label):

        nplabel = np.empty((2, 16, 352, 512))
        t1label = label.copy()
        t1label[label == 2] = 1 # 肝脏
        t2label = label.copy()
        t2label[label == 1] = 0 # 病灶
        t2label[label == 2] = 1 # 病灶
        nplabel[0, :, :, :] = t1label# 肝脏
        nplabel[1, :, :, :] = t2label# 病灶
        return nplabel

    def get_train_data(self,frameNum, resize_scale=1):
        tmpData = []
        length = len(self.filename_list)

        for ctl in range(length):
            img, label = self.get_np_data_3d(self.filename_list[ctl],resize_scale=resize_scale)#c y x

            img = self.clamp(img)#
            i = 0
            tmpNum = img.shape[0] - frameNum
            while i < tmpNum:
                tc = Counter(label[i, :, :].flatten())
                if tc[2] > 10: #Tumor的帧
                    tlabel = label[i:i+frameNum,:,:]
                    timg = img[i:i+frameNum,:,:]

                    timg1, tlabel1 = self.randomCrop(timg, tlabel, 70, 90)#随机裁剪1
                    nplabel = self.seperateLiverTumorMask(tlabel1)
                    tmpData.append([np.expand_dims(timg1 ,axis=0), nplabel])

                    # timg11, tlabel11 = self.fold(timg1, tlabel1, 1) #上下翻折
                    # nplabel = self.seperateLiverTumorMask(tlabel11)
                    # tmpData.append([np.expand_dims(timg11 ,axis=0), nplabel])

                    i += frameNum
                else:
                    i += 1
        return tmpData

    def get_np_data_3d(self, filename, resize_scale=1):
        data_np = sitk_read_raw(self.dataset_path + '/data/' + filename,
                                resize_scale=resize_scale)
        label_np = sitk_read_raw(self.dataset_path + '/label/' + filename.replace('volume', 'segmentation'),
                                 resize_scale=resize_scale)
        return data_np, label_np
'''

# 测试代码
if __name__ == '__main__':
    """
    fixed_path  = '/home/data/LiTS/fixed_data'
    dataset = Lits_DataSet(16,1,fixed_path,mode='val')  #batch size
    # dataset = Lits_DataSet(16,1,fixed_path,mode='train') 
    data_loader=DataLoader(dataset=dataset,batch_size=1,num_workers=1, shuffle=False)
    # for batch_idx, (data, target, fullImg) in enumerate(data_loader):
    print(len(data_loader))
    for batch_idx, (data, target) in enumerate(data_loader):
        # target = to_one_hot_3d(target.long())
        print(data.shape, target.shape, torch.unique(target))
        print(data.dtype,target.dtype)
        assert 1>3
    """
    args = {}

    # img_paths = glob('/home/jzw/data/LiTS/LITS17/train_image_352*352/*')
    # img_paths = glob('/home/jzw/data/LiTS/LITS17/train_image2d/*')
    img_paths = glob('/home/jzw/data/LiTS/LITS17/train_image_352*352_nospacing/*')
    # for p in img_paths:
    #     print(Path(p).stem)
    #     assert 1>2
    img_paths = [p for p in img_paths if "slice-130" in Path(p).stem]
    # print(len(img_paths))
    # assert 1>4

    dataset = LitsDataset(args, img_paths)
    # dataset = LitsDataSet(args, img_root, mask_root)
    data_loader=DataLoader(dataset=dataset, batch_size=1, num_workers=0, shuffle=False)
    # for batch_idx, (data, target, fullImg) in enumerate(data_loader):
    # device = torch.device("cuda:0")

    # visual_directory = Path("./dataset/lits/visual/train_image_352*352_nospacing")
    visual_directory = Path("./visual/train_image_352*352_nospacing")
    visual_directory.mkdir(parents=True, exist_ok=True) # ----------------------------------<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # print(visual_directory)
    # assert 1>4

    for batch_idx, (image, mask, img_path) in enumerate(data_loader):
        # print(image.shape, mask.shape)
        # print(img_path[0])
        img_name = Path(img_path[0]).stem
        img_num = img_name.split("-")[1]
        # print(img_name, img_num)
        # if batch_idx > 2:
        #     break
        # image = image.to(device)
        # mask = mask.to(device)
        
        # target = to_one_hot_3d(target.long())
        # print(f"image: {image.shape}, {image.min()}, {image.max()}, {image.mean()}")
        # print(f"mask: {mask.shape}, {mask.min()}, {mask.max()}, {mask.mean()}")

        ## 可视化img和mask
        saved_img = (image[0][:1].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
        # print(saved_img.min(), saved_img.max(), saved_img.mean())
        cv2.imwrite(str(visual_directory / f"{img_num}_img.png"), saved_img)
        saved_mask0 = (mask[0][:1].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
        # print(np.unique(saved_mask0))
        cv2.imwrite(str(visual_directory / f"{img_num}_liver_mask.png"), saved_mask0)
        saved_mask1 = (mask[0][1:2].permute(1, 2, 0) * 255.0).cpu().numpy().astype(np.uint8)
        # print(np.unique(saved_mask1))
        cv2.imwrite(str(visual_directory / f"{img_num}_tumor_mask.png"), saved_mask1)