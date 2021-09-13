import sys
sys.path.append("/home/lpy/paper/LiTS/dataset/")
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch,os
from torch.utils.data import Dataset, DataLoader
import cv2


class Lits_DataSet(Dataset):
    def __init__(self, crop_size,resize_scale, dataset_path,mode=None):
        self.crop_size = crop_size
        self.resize_scale=resize_scale
        self.dataset_path = dataset_path
        # self.n_labels = 3

        if mode=='train':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'train_name_list.txt'))
        elif mode =='val':
            self.filename_list = load_file_name_list(os.path.join(dataset_path, 'val_name_list.txt'))
        else:
            raise TypeError('Dataset mode error!!! ')


    def __getitem__(self, index):
        data, target, edge = self.get_train_batch_by_index(crop_size=self.crop_size, index=index,
                                                     resize_scale=self.resize_scale)
        return torch.from_numpy(data), torch.from_numpy(target), torch.from_numpy(edge)

    def __len__(self):
        return len(self.filename_list)

    def clamp(self, niiImage):
        ans = niiImage.copy()
        ans[ans < -200] = -200
        ans[ans > 250] = 250
        maxx = np.max(ans)
        minn = np.min(ans)
        # ansMean = ans.mean()
        # ansStd =  ans.std()
        maxx = maxx + 1 if maxx == minn else maxx # 如果minn和maxx相等，为了不除0，将maxx加1
        return 2 * (ans - minn) / (maxx - minn) - 1

    def sobel(self, niiImage):
        ans = np.zeros_like(niiImage)# channel, w, h
        for i in range(niiImage.shape[0]):
            maxx = max(1, np.max([niiImage[i, :, :]]))
            tmp = niiImage[i, :, :] / maxx
            tmp = np.uint8(tmp * 255)
            xedge = cv2.Sobel(tmp, cv2.CV_16S, 1, 0, ksize=1)
            xedge = cv2.convertScaleAbs(xedge)
            yedge = cv2.Sobel(tmp, cv2.CV_16S, 0, 1, ksize=1)
            yedge = cv2.convertScaleAbs(yedge)
            edge = xedge + yedge
            edge[edge > 0] = 255
            ans[i, :, :] = edge // 255
        ans = ans.astype(np.uint8)
        return ans

    def get_train_batch_by_index(self,crop_size, index,resize_scale=1):
        img, label = self.get_np_data_3d(self.filename_list[index],resize_scale=resize_scale)
        img, label = random_crop_3d(img, label, crop_size)
        img = self.clamp(img) 
        # print('label',label.shape)
        # print(np.unique(label)) # [0, 1, 2]
        labelShape = label.shape
        nplabel = np.empty((2, labelShape[0], labelShape[1], labelShape[2]))
        t1label = label.copy()
        t1label[label == 1] = 1 # 肝脏
    
        t2label = label.copy()
        t2label[label == 2] = 1 # 病灶
        
        nplabel[0, :, :, :] = t1label
        nplabel[1, :, :, :] = t2label
        npedge = np.zeros_like(nplabel)
        npedge[0, :, :, :] = self.sobel(t1label)
        npedge[1, :, :, :] = self.sobel(t2label)

        return np.expand_dims(img,axis=0), nplabel, npedge # 2, channel, w, h

    def get_np_data_3d(self, filename, resize_scale=1):
        data_np = sitk_read_raw(self.dataset_path + '/data/' + filename,
                                resize_scale=resize_scale)
        label_np = sitk_read_raw(self.dataset_path + '/label/' + filename.replace('volume', 'segmentation'),
                                 resize_scale=resize_scale)
        return data_np, label_np

# 测试代码
    
if __name__ == '__main__':
    fixd_path  = '/home/data/LiTS/fixed_data'
    dataset = Lits_DataSet([48, 256, 256],1,fixd_path,mode='train')  #batch size
    data_loader=DataLoader(dataset=dataset,batch_size=2,num_workers=1, shuffle=True)
    for batch_idx, (data, target, edge) in enumerate(data_loader):
        # target = to_one_hot_3d(target.long())
        print(data.shape, target.shape,edge.shape, torch.max(data), torch.min(data))
        # assert 1>3