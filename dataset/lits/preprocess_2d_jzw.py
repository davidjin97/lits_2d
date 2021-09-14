import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
from skimage.transform import resize
import time
import sys
sys.path.append('/home/jzw/workspace_seg/template_seg_jzw/')
sys.path.append('/home/jzw/workspace_seg/template_seg_jzw/dataset/')
from utils.utils import *
import zipfile
from pathlib import Path


def normalize(slice, bottom=99.5, down=0.5):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        # 但我发现强度并不是在0-5000
        tmp[tmp == tmp.min()] = -9
        return tmp

def generate_subimage(ct_array, seg_array, stridez, stridex, stridey, blockz, blockx, blocky,
					  idx, origin, direction, xyz_thickness, savedct_path, savedseg_path, ct_file):
    num_z = (ct_array.shape[0]-blockz)//stridez + 1 # math.floor()
    num_x = (ct_array.shape[1]-blockx)//stridex + 1
    num_y = (ct_array.shape[2]-blocky)//stridey + 1

    for z in range(num_z):
        for x in range(num_x):
            for y in range(num_y):
                seg_block = seg_array[z*stridez:z*stridez+blockz,x*stridex:x*stridex+blockx,y*stridey:y*stridey+blocky]
                if seg_block.any():
                        ct_block = ct_array[z * stridez:z * stridez + blockz, x * stridex:x * stridex + blockx,
                                        y * stridey:y * stridey + blocky]
                        saved_ctname = os.path.join(savedct_path,'volume-'+str(idx) +'.npy')
                        saved_segname = os.path.join(savedseg_path,'segmentation-'+str(idx)+'.npy')
                        np.save(saved_ctname, ct_block)
                        np.save(saved_segname, seg_block)
                        idx = idx + 1
    return idx

def generate_subimage_224_224(ct_array, seg_array, stridez, stridex, stridey, blockz, blockx, blocky,
					  idx, origin, direction, xyz_thickness, savedct_path, savedseg_path, ct_file):
    print(ct_array.shape)
    sz, sx, sy = ct_array.shape[0], ct_array.shape[1], ct_array.shape[2]
    num_z = (sz-blockz)//stridez + 1 # math.floor()
    num_x = (sx-blockx)//stridex + 1
    num_y = (sy-blocky)//stridey + 1
    # print(num_z, num_x, num_y)

    for z in range(num_z):
        nxy = 0
        for x in range(num_x+1):
            ny = 0
            for y in range(num_y+1):
                stz, stx, sty = z*stridez, x*stridex, y*stridey
                if x == num_x: # x向最后一个
                    stx = sx-blockx
                    if stx == (num_x-1)*stridex:
                        continue
                if y == num_y: # y向最后一个
                    sty = sy-blocky
                    if sty == (num_y-1)*stridey:
                        continue
                seg_block = seg_array[stz: stz+blockz, stx: stx+blockx, sty: sty+blocky]
                # print(seg_block.min(), seg_block.max())
                if seg_block.any():
                    ct_block = ct_array[stz: stz+blockz, stx: stx+blockx, sty: sty+blocky]
                    # print(seg_block.shape, ct_block.shape)
                    saved_ctname = savedct_path / ('slice-' + str(idx) + '.npy')
                    saved_segname = savedseg_path / ('seg-' + str(idx) + '.npy')
                    np.save(str(saved_ctname), ct_block)
                    np.save(str(saved_segname), seg_block)
                    idx = idx + 1
                ny += 1
            # print("ny: ", ny)
            # nxy += ny
        # print("nxy: ", nxy)
        # assert 1>5
    return idx

def get_realfactor(spa,xyz,ct_array):
    # spa = [0.84375 0.84375 1.5    ], xyz = [0.8, 0.8, 1.5]
    resize_factor = spa / xyz # 还要再颠一下顺序，再确定缩放后像素个数的整数
    # print("resize_factor", resize_factor)
    new_real_shape = ct_array.shape * resize_factor[::-1] # zyx
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / ct_array.shape
    # print("real_resize_factor", real_resize_factor)
    return real_resize_factor

def unzip(filename, dstdir):
    with zipfile.ZipFile(filename,'r') as f:
        f.extractall(dstdir)
    os.remove(filename)

def find_bb(volume):
    """
    zyx轴只取含有liver/liver tumor的帧
    """
    img_shape = volume.shape
    bb = np.zeros((6,), dtype=np.uint)
    bb_extend = 30
    bb_extend_axis = 0 # axis方向不扩展
    # axis
    for i in range(img_shape[0]):
        img_slice_begin = volume[i,:,:]
        if np.sum(img_slice_begin)>0:
            bb[0] = np.max([i-bb_extend_axis, 0]) # axis方向不扩展
            break

    for i in range(img_shape[0]):
        img_slice_end = volume[img_shape[0]-1-i,:,:]
        if np.sum(img_slice_end)>0:
            bb[1] = np.min([img_shape[0]-1-i + bb_extend_axis, img_shape[0]-1]) # axis方向不扩展
            break
    # seg
    for i in range(img_shape[1]):
        img_slice_begin = volume[:,i,:]
        if np.sum(img_slice_begin)>0:
            bb[2] = np.max([i-bb_extend, 0])
            break

    for i in range(img_shape[1]):
        img_slice_end = volume[:,img_shape[1]-1-i,:]
        if np.sum(img_slice_end)>0:
            bb[3] = np.min([img_shape[1]-1-i + bb_extend, img_shape[1]-1])
            break

    # coronal
    for i in range(img_shape[2]):
        img_slice_begin = volume[:,:,i]
        if np.sum(img_slice_begin)>0:
            bb[4] = np.max([i-bb_extend, 0])
            break

    for i in range(img_shape[2]):
        img_slice_end = volume[:,:,img_shape[2]-1-i]
        if np.sum(img_slice_end)>0:
            bb[5] = np.min([img_shape[2]-1-i+bb_extend, img_shape[2]-1])
            break
	
    return bb

def preprocess():
    """
    get 3d npy image, size_zyx: [64, 128, 160]
    """
    start_time = time.time()
    ##########hyperparameters1##########
    images_path = Path('/home/jzw/data/LiTS/LITS17/CT')
    labels_path = Path('/home/jzw/data/LiTS/LITS17/seg')
    if not images_path.exists():
        print("images_path doesn't exist")
    if not labels_path.exists():
        print("labels_path doesn't exist")

    savedct_path = Path('/home/jzw/data/LiTS/LITS17/train_image_224*224')
    savedseg_path = Path('/home/jzw/data/LiTS/LITS17/train_mask_224*224')

    train_image = savedct_path
    train_mask = savedseg_path
    train_image.mkdir(exist_ok=True)
    print("trainImage 预处理后目录创建成功")
    train_mask.mkdir(exist_ok=True)
    print("trainMask 预处理后输出目录创建成功")

    # 处理训练数据
    saved_idx = 0
    expand_slice = 10
    new_spacing = [0.8, 0.8, 1.5]
    # blockz = 64; blockx = 128; blocky = 160 # 每个3d patch的大小
    # stridez = blockz//6; stridex = blockx//4; stridey = blocky//3
    blockz = 1; blockx = 224; blocky = 224# 每个2d patch的大小
    stridez = 1; stridex = blockx//4; stridey = blocky//4
    for ct_file in images_path.iterdir():
        # print(ct_file) # /home/jzw/data/LiTS/LITS17/CT/volume-119.nii
        ct = sitk.ReadImage(str(ct_file), sitk.sitkInt16) # sitk.sitkInt16 Read one image using SimpleITK, <class 'SimpleITK.SimpleITK.Image'>
        origin = ct.GetOrigin() # <class 'tuple'> (-214.578125, -381.578125, -598.5)
        direction = ct.GetDirection() # <class 'tuple'> (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        spacing = np.array(list(ct.GetSpacing())) # [0.84375 0.84375 1.5    ]
        
        ct_array = sitk.GetArrayFromImage(ct) # (461, 512, 512)
        seg = sitk.ReadImage(str(labels_path / ct_file.name.replace('volume', 'segmentation')), sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg) # (461, 512, 512))

        # step1: spacing interpolation
        real_resize_factor = get_realfactor(spacing, new_spacing, ct_array) # [1.        1.0546875 1.0546875]
        # 根据输出out_spacing设置新的size
        # nearest is order=0, Bilinear interpolation would be order=1, and cubic is the default (order=3).
        ct_array = ndimage.zoom(ct_array, real_resize_factor, order=3) 
        # 对gt插值不应该使用高级插值方式，这样会破坏边界部分,检查数据输出很重要！！！
        # 使用order=0 nearst差值可确保zoomed seg unique = [0,1,2]
        seg_array = ndimage.zoom(seg_array, real_resize_factor, order=0)
        # print('new space', new_spacing) # [0.8, 0.8, 1.5]
        # print('zoomed shape:', ct_array.shape, ',', seg_array.shape) # (461, 540, 540) , (461, 540, 540)

        # step2 :get mask effective range(startpostion:endpostion) z轴只取含有liver/liver tumor的帧
        pred_liver = seg_array.copy()
        pred_liver[pred_liver>0] = 1
        bb = find_bb(pred_liver) # liver和tumor都是liver
        ct_array = ct_array[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]]
        seg_array = seg_array[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]]
        # print('effective shape:', ct_array.shape,',',seg_array.shape) # (137, 257, 283) , (137, 257, 283)

        # step3:标准化Normalization
        ct_array_nor = normalize(ct_array)

        # 切割原图，产生多个子图 
        if ct_array.shape[0] < blockz:
            print('generate no subimage !')
        else:
            saved_idx = generate_subimage_224_224(ct_array_nor, seg_array,stridez, stridex, stridey, blockz, blockx, blocky,
                            saved_idx, origin, direction, new_spacing, savedct_path, savedseg_path, ct_file)

        print('Time {:.3f} min'.format((time.time() - start_time) / 60))
        print(saved_idx)

if __name__ == '__main__':
    start_time = time.time()
    # logfile = '../logs/printLog0117'
    # if os.path.isfile(logfile):
    #     os.remove(logfile)
    # sys.stdout = Logger(logfile)#see utils.py
	##########hyperparameters##########
    preprocess()

	# Decide preprocess of different stride and window
	# Decide_preprocess(blockzxy,config)

    print('Time {:.3f} min'.format((time.time() - start_time) / 60))
    print(time.strftime('%Y/%m/%d-%H:%M:%S', time.localtime()))


    # ## test the saved np
    # ### 3d images
    # # savedct_path = '/home/jzw/data/LiTS/LITS17/train_image3d_jzw/*'
    # # savedseg_path = '/home/jzw/data/LiTS/LITS17/train_mask3d_jzw/*'
    # savedct_path = '/home/jzw/data/LiTS/LITS17/train_image3d/*'
    # savedseg_path = '/home/jzw/data/LiTS/LITS17/train_mask3d/*'
    # ### 2d images
    # # savedct_path = '/home/jzw/data/LiTS/LITS17/train_image2d/*'
    # # savedseg_path = '/home/jzw/data/LiTS/LITS17/train_mask2d/*'
    # from glob import glob
    # ct_paths = glob(savedct_path)
    # print(len(ct_paths))
    # for path in ct_paths[:20]:
    #     x = np.load(path) 
    #     print(type(x), x.shape)