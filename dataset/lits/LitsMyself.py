import sys
from glob import glob
import logging
sys.path.append("/home/jzw/workspace/lits_2d/dataset/lits")
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch,os
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from pathlib import Path
logger = logging.getLogger('global')

class Lits_DataSet(Dataset):
    """
    通过`train_name_list.txt`与`val_name_list.txt`预先划分104例train volume与27例test volume
    值得注意的是：所使用的`fixed_data`目录下的.nii数据，是经过了部分图像级图处理操作的，预处理文件为：`preprocess2.py`
    """
    def __init__(self, frameNum, resize_scale, dataset_path,mode=None):
        self.frameNum = frameNum
        self.resize_scale=resize_scale
        self.dataset_path = Path(dataset_path)

        if mode=='train':
            self.filename_list = load_file_name_list(str(self.dataset_path / 'train_name_list.txt'))
        elif mode =='test':
            self.filename_list = load_file_name_list(str(self.dataset_path / 'val_name_list.txt'))
        elif mode =='debug':
            self.filename_list = load_file_name_list(str(self.dataset_path / 'debug_name_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')

        print('the length of filename list: ', len(self.filename_list))
        # self.data = self.get_data(self.frameNum, 1)
        self.generate_dir = self.dataset_path / f"fixed_{mode}_{frameNum}_{resize_scale}"
        if not self.generate_dir.exists():
            self.get_data_and_write(self.frameNum, 1)
        # self.get_data_and_write(self.frameNum, 1)

        self.img_paths = glob(str(self.generate_dir / "img" / "*"))
        self.label_paths = list(map(lambda x: x.replace('img', 'label'), self.img_paths))

    def __getitem__(self, index):
        # img, target = self.data[index][0], self.data[index][1]
        # return torch.from_numpy(img), torch.from_numpy(target)
        img_path = self.img_paths[index]
        label_path = self.label_paths[index]
        npimg = np.load(img_path)
        nplabel = np.load(label_path) # 0表示背景，1表示肝脏，2表示肝脏肿瘤
        # print(npimg.shape, nplabel.shape) # (1, 6, 512, 512) -- (c, d, h, w)
        return torch.from_numpy(npimg), torch.from_numpy(nplabel)

    def __len__(self):
        # return len(self.data)
        return len(self.img_paths)

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

    def seperate_liver_tumor_mask(self, label):
        ## 读取的label：0为背景，1为肝脏，2为肿瘤
        label_1 = label.copy()
        label_1[label == 2] = 1 # 将tumor填充为liver
        label_2 = label.copy()
        label_2[label == 1] = 0 # 将liver设为背景
        label_2[label == 2] = 1 # 仅保留tumor为mask

        label_shape = label.shape
        nplabel = np.empty((2, label_shape[0], label_shape[1], label_shape[2]))
        nplabel[0, :, :, :] = label_1 # 肝脏
        nplabel[1, :, :, :] = label_2 # 病灶
        return nplabel

    def normalize_2d(self, volume, top=99.5, bottom=0.5):
        """
        normalize image with mean and std for regionnonzero,and clip the value into range
        :param volume: [c, d ,h ,w]
        """
        slice_num = volume.shape[1]
        volume = volume.astype(np.float64)
        for si in range(slice_num):
            slice = volume[:, si, :, :]
            slice_nonzero = slice[np.nonzero(slice)]
            volume[:, si, :, :] = (slice - np.mean(slice_nonzero)) / (np.std(slice_nonzero))
            
        # print(volume.dtype, volume.min(), volume.max(), volume.mean(), volume.std())
        return volume

    def normalize_3d(self, volume, top=99.5, bottom=0.5):
        """
        normalize image with mean and std for regionnonzero,and clip the value into range
        :param volume: shape [c, d, h, w]
        """
        # t = np.percentile(volume, top)
        # b = np.percentile(volume, bottom)
        # volume = np.clip(volume, t, b)
        volume_nonzero = volume[np.nonzero(volume)]
        # if np.std(volume) == 0 or np.std(volume_nonzero) == 0:
        #     return volume 
        # else:
        tmp = (volume - np.mean(volume_nonzero)) / (np.std(volume_nonzero))
        tmp = tmp.astype(np.float32)
        return tmp

    def get_data(self, frameNum, resize_scale=1):
        """
        从filename_list中加载每一个volume的数据，
        所返回的data为frameNum帧的3d数据，其中image做了normalize，而label到做了one-hot的变化
        """
        data_list = []
        length = len(self.filename_list)
        for volume_id in range(length):
            print('loading volume: ', self.filename_list[volume_id])
            logger.info(f'loading volume: {self.filename_list[volume_id]}')
            img, label = self.get_np_volume_pair(self.filename_list[volume_id], resize_scale=resize_scale)
            for i in range(0, img.shape[0]-frameNum+1, frameNum):
                tlabel = label[i: i+frameNum, :, :]
                timg = img[i: i+frameNum, :, :]

                nplabel = self.seperate_liver_tumor_mask(tlabel)
                npimg = np.expand_dims(timg ,axis=0)

                npimg = self.normalize_3d(npimg)

                data_list.append([npimg, nplabel])
        return data_list

    def get_data_and_write(self, frameNum, resize_scale=1):
        """
        从filename_list中加载每一个volume的数据，
        所返回的data为frameNum帧的3d数据，其中image没有再进行额外的处理，而label到做了one-hot的变化
        """
        length = len(self.filename_list)
        savedct_path = self.generate_dir / "img"
        savedseg_path = self.generate_dir / "label"
        savedct_path.mkdir(parents=True, exist_ok=True)
        savedseg_path.mkdir(parents=True, exist_ok=True)
        slice_nums = 0
        for volume_id in range(length):
            print('loading volume: ', self.filename_list[volume_id])
            logger.info(f'loading volume: {self.filename_list[volume_id]}')
            img, label = self.get_np_volume_pair(self.filename_list[volume_id], resize_scale=resize_scale)
            for i in range(0, img.shape[0]-frameNum+1, frameNum):
                tlabel = label[i: i+frameNum, :, :]
                timg = img[i: i+frameNum, :, :]

                nplabel = self.seperate_liver_tumor_mask(tlabel)
                npimg = np.expand_dims(timg ,axis=0)

                npimg = self.normalize_3d(npimg)
  
                ## 对2d数据集的特殊处理
                if self.frameNum == 1:
                    npimg = npimg[:, 0, :, :]
                    nplabel = nplabel[:, 0, :, :]

                saved_ctname = savedct_path / (f"{slice_nums}_{self.filename_list[volume_id].split('.')[0].split('-')[1]}_{i}.npy")
                saved_segname = savedseg_path / (f"{slice_nums}_{self.filename_list[volume_id].split('.')[0].split('-')[1]}_{i}.npy")
                np.save(str(saved_ctname), npimg)
                np.save(str(saved_segname), nplabel)

                slice_nums += 1
        # return data_list

    def get_np_volume_pair(self, filename, resize_scale=1):
        """
        load one volume pair(data, label) as numpy array
        return:
            image and label with shape: (c, h, w)
        """
        data_np = sitk_read_raw(str(self.dataset_path / 'data' / filename),
                                resize_scale=resize_scale)
        label_np = sitk_read_raw(str(self.dataset_path / 'label' / filename.replace('volume', 'segmentation')),
                                resize_scale=resize_scale)
        return data_np, label_np

# 测试代码
    
if __name__ == '__main__':
    fixd_path  = '/home/jzw/data/LiTS/fixed_data'
    dataset = Lits_DataSet(1, 1, fixd_path, mode='test')  #batch size
    data_loader=DataLoader(dataset=dataset, batch_size=1, num_workers=1, shuffle=False)
    # for batch_idx, (data, target, fullImg) in enumerate(data_loader):
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to("cuda:2")
        target = target.to("cuda:2")

        if batch_idx < 10:
            print(data.shape, target.shape, torch.unique(target))
            print(data.min().data, data.max().data, data.mean().data, data.std().data) 
        else:
            break
        # 2 1 16 512 512   2 2 16 512 512