import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import torch

def load_file_name_list(file_path):
    """
    加载保存train或test所使用的volume名字
    """
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
            file_name_list.append(lines)
    return file_name_list

def sitk_read_raw(img_path, resize_scale=1):
    """
    读取3D图像并rescale（因为一般医学图像并不是标准的[1,1,1]scale）
    return:
        3d volume with shape: (d, h, w)
    """
    nda = sitk.ReadImage(img_path)
    if nda is None:
        raise TypeError(f"input img [{img_path}] is None!!!")
    nda = sitk.GetArrayFromImage(nda)  # channel first
    # rescale resize函数，order=0表示最近邻插值
    nda = ndimage.zoom(nda,[resize_scale,resize_scale,resize_scale],order=0)

    return nda