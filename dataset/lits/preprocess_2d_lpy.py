'''
    Func:制作训练集
'''
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import cv2

ct_name = ".nii"
mask_name = ".nii"

ct_path = '/home/jzw/data/LiTS/LITS17/CT'
seg_path = '/home/jzw/data/LiTS/LITS17/seg'

outputImg_path = "/home/jzw/data/LiTS/LITS17/train_image2d"
outputMask_path = "/home/jzw/data/LiTS/LITS17/train_mask2d"

if not os.path.exists(outputImg_path):
    os.mkdir(outputImg_path)
if not os.path.exists(outputMask_path):
    os.mkdir(outputMask_path)

def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files

def crop_ceter(img, croph, cropw):
    #for n_slice in range(img.shape[0]):
    height, width = img[0].shape
    starth = height//2 - (croph//2)
    startw = width//2 - (cropw//2)
    return img[:, starth:starth+croph, startw:startw+cropw]

if __name__ == "__main__":

    for index, file in enumerate(tqdm(os.listdir(ct_path))):
        print(os.path.join(ct_path, file))

        # 获取CT图像及Mask数据
        ct_src = sitk.ReadImage(os.path.join(ct_path, file), sitk.sitkInt16)
        mask = sitk.ReadImage(os.path.join(seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
        # GetArrayFromImage()可用于将SimpleITK对象转换为ndarray
        ct_array = sitk.GetArrayFromImage(ct_src)
        mask_array = sitk.GetArrayFromImage(mask)

        # mask_array[mask_array == 1] = 0  # 肿瘤
        # mask_array[mask_array == 2] = 1

        # 阈值截取
        ct_array[ct_array > 200] = 200
        ct_array[ct_array < -200] = -200

        ct_array = ct_array.astype(np.float32)
        ct_array = ct_array / 200

        # 找到肝脏区域开始和结束的slice，并各向外扩张slice
        z = np.any(mask_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]
        assert 1>4

        ct_crop = ct_array[start_slice-1:end_slice + 2, :, :]
        mask_crop = mask_array[start_slice:end_slice + 1, :, :]

        #裁剪(偶数才行) 448*448
        ct_crop = ct_crop[:,32:480,32:480]
        mask_crop = mask_crop[:,32:480,32:480]

        print('ct_crop.shape',ct_crop.shape)

        # 切片处理,并去掉没有病灶的切片
        for n_slice in range(mask_crop.shape[0]):
            maskImg = mask_crop[n_slice, :, :]
            ctImageArray = np.zeros((ct_crop.shape[1], ct_crop.shape[2], 3), np.float)
            ctImageArray[:, :, 0] = ct_crop[n_slice , :, :]
            ctImageArray[:, :, 1] = ct_crop[n_slice + 1, :, :]
            ctImageArray[:, :, 2] = ct_crop[n_slice + 2, :, :]

            imagepath = outputImg_path + "/" + str(index+1) + "_" + str(n_slice) + ".npy"
            maskpath = outputMask_path + "/" + str(index+1) + "_" + str(n_slice) + ".npy"
                
            np.save(imagepath, ctImageArray)  # (448，448,3) np.float dtype('float64')
            np.save(maskpath, maskImg)  # (448，448) dtype('uint8') 值为0 1 2
    print("Done！")
