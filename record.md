# 0913
显存分析
batch=2
size = [2, 1, 64, 128, 160]，显存在epoch=2的第2个iter会爆
2*1*64*128*160 = 2621440

# 0914
执行224*224数据切割
nohup python preprocess_2d_jzw.py > preprocess_2d_jzw_224.log 2>&1 &
pid = 48106
376649条数据

执行448*448数据切割
nohup python preprocess_2d_jzw.py > preprocess_2d_jzw_448.log 2>&1 &
pid = 3217
13700条数据
nohup python preprocess_2d_jzw.py > preprocess_2d_jzw_448_tmp.log 2>&1 &
pid = 23139
条数据

# 0915
512 * 512通过spacing的统一后，最小分辨率变为356 * 356，因此448裁剪失败
可以尝试352 * 352
执行352 * 352数据切割
nohup python preprocess_2d_jzw.py > preprocess_2d_jzw_352.log 2>&1 &
pid = 20692

Q: loss is negative?
A: use `sigmoid` at the end of model.

run lits_352*353 by UNet
nohup bash run.sh > train_lits_unet.out 2>&1 &
pid = 34517

nohup bash run.sh > train_lits_unet_epochloss.out 2>&1 &
pid = 7854

# 0916
add ious and dices index in trainer

Q: why dice coef > 1?
A: the process in LitsDataset has bug
nohup bash run.sh > train_lite_unet_loss-iou-dice.out 2>&1 &
pid = 13591

M: perfect the output system

train ResUNet and UNet
CUDA_VISIBLE_DEVICES=0,1 python main.py --config Unet.yaml --world_size 2 --logname train_lits_unet_runsout.log
nohup bash run.sh > train_lits_unet_runsout.out 2>&1 &
pid = 6812
CUDA_VISIBLE_DEVICES=2,3 python main.py --config ResUnet.yaml --world_size 2 --logname train_lits_resunet.log
nohup bash run.sh > train_lits_resunet.out 2>&1 &
pid = 7075

# 0922
编写test方法

# 0924
在test中将结果进行可视化，发现tumor的mask有些分散，并且没有成功预测

add model `SEUnet`
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEUnet.yaml --world_size 2 --logname train_lits_seunet.log
nohup bash run.sh > train_lits_seunet.out 2>&1 &
pid = 20756

使用 train_image2d预处理数据进行计算
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEUnet.yaml --world_size 2 --logname train_lits_train_image2d_seunet.log
tensorboard --logdir runs/lits_seg/seunet/train/2021-09-24-16-23-49/summarys

尝试只分割tumor
CUDA_VISIBLE_DEVICES=2,3 python main.py --config SEUnet.yaml --world_size 2 --logname train_lits_tumor_seunet.log
tensorboard --logdir runs/lits_seg/seunet/train/2021-09-24-22-11-10/summarys
loss -> 0.6931, iou -> 0.163, dice ~ 0

# 0925
阅读路师姐的模型代码

# 0926
使用focal loss训练
CUDA_VISIBLE_DEVICES=3 python main.py --config SEUnet.yaml --world_size 1 --logname train_lits_seunet_focal.log

# 0927
玩明白focal loss的使用

查看mask是否有问题
`preprocess_2d_jzw.py`
保存npy时记录slice在原volume中是第多少帧
做一个`train_image_352*352_nospacing`，没有进行spacing操作，也没有将最小值化为-9
nohup python preprocess_2d_jzw.py > preprocess_2d_jzw_352_nospacing_no-9_slicenum.log 2>&1 &
对`train_image_352*352_nospacing`数据进行可视化检查
1. 在`preprocess_2d_jzw.py`中直接测试保存的.npy是否与原图一致，以volume-130可视化为例

做一个`train_image_352*352_no-9`，及上述出去no-spacing操作，其余保持一致
nohup python preprocess_2d_jzw.py > preprocess_2d_jzw_352_no-9.log 2>&1 &

# 0928
继续昨日可视化的测试
1. 对`train_image_352*352_nospacing`进行dataset的可视化检测，
图片存入`./dataset/lits/visual/train_image_352*352_nospacing`，没有问题
使用该预处理数据，使用SENet训练
CUDA_VISIBLE_DEVICES=3 python main.py --config SEUnet.yaml --world_size 1 --logname train_lits_seunet_bcel.log
nohup bash run.sh > train_lits352nospacing_seunet_bcel.out 2>&1 &
pid = 1639, 2021-09-28-09-08-42

2. 对`train_image_352*352_no-9`保存的.npy可视化

对于dice不正常问题，检查dice函数是否有问题
python metric.py
dice_coef好像也没有问题
再看一下summary
tensorboard --logdir runs/lits_seg/seunet/train/2021-09-24-16-23-49/summarys
dice_2不正常
tensorboard --logdir runs/lits_seg/seunet/train/2021-09-24-22-11-10/summarys
单独train tumor dice也是不正常的

# 0929
回顾昨日工作
查看 2021-09-28-09-08-42, train_lits_seunet_bcel 结果
tensorboard --logdir runs/lits_seg/seunet/train/2021-09-28-09-08-42/summarys
依旧是dice_1正常，且很高，但dice_2非常的低
需要对train好的模型进行test，并且可视化output结果
CUDA_VISIBLE_DEVICES=3 python main.py --config SEUnet.yaml --world_size 1 -e
> 2021-09-29-14-37-09 可视化了40个iter的结果
> 可以看出大部分tumor的output都是全0的
> 2021-09-29-15-00-44 表明几乎所有tumor的output都是0
> 2021-09-29-15-21-22 查看了output的最大最小值
> tumor 输出的概率值就是很低，且发现tumor moiu还挺正常，但dice不行
> 2021-09-29-15-32-21 查看iou_score
> tumor的iou_score正常，但dice很低

# 1010
同3d模型一样，修复metric的计算
CUDA_VISIBLE_DEVICES=3 python main.py --config config/Unet.yaml --world_size 1 --logname train_lits_unet_bcel_debugmetric.log
nohup bash run.sh > train_lits_unet_bcel_debugmetric.out 2>&1 &
此次使用数据 `train_image_352*352_nospacing`，最终应该使用`train_image_352*352_no-9`

CUDA_VISIBLE_DEVICES=2 python main.py --config config/Unet.yaml --world_size 1 --logname train_lits_unet_focal_debugmetric.log
nohup bash run.sh > train_lits_unet_focal_debugmetric.out 2>&1 &

## 查看训练结果
tensorboard --logdir runs/lits_seg/unet/train/2021-09-16-22-20-37/summarys
tensorboard --logdir runs/lits_seg/unet/train/2021-10-10-13-26-34/summarys
tensorboard --logdir runs/lits_seg/unet/train/2021-10-10-13-27-23/summarys
今天两次跑的liver_dscs一直都是0.2424

## debug train
CUDA_VISIBLE_DEVICES=1 python main.py --config config/Unet.yaml --world_size 1 --logname train_lits_unet_debug.log
nohup bash run.sh > train_lits_unet_debug.out 2>&1 &
# 1011
求指标时两次使用了sigmoid
tensorboard --logdir runs/lits_seg/unet/train/2021-10-10-21-58-19/summarys

## 修复sigmoid问题以后train unet bcel
CUDA_VISIBLE_DEVICES=1 python main.py --config config/Unet.yaml --world_size 1 --logname train_lits_unet_bcel_debugsigmoid.log --debug
CUDA_VISIBLE_DEVICES=1 python main.py --config config/Unet.yaml --world_size 1 --logname train_lits_unet_bcel_debugsigmoid.log
nohup bash run.sh > train_lits_unet_bcel_debugsigmoid.out 2>&1 &
**查看两个epoch的dsc是否一致？**
结果：不一致，这是合理的。
tensorboard --logdir runs/lits_seg/unet/train/2021-10-11-09-32-19/summarys
epoch9: train_dsc正常上升至0.9482，loss降至0.666稳住了，tumor_dsc一直在0.6244
epoch20: train_dsc正常上升至0.9676，loss降至0.666稳住了，tumor_dsc一直在0.6244

## 查看seunet
tensorboard --logdir runs/lits_seg/seunet/train/2021-09-28-09-08-42/summarys
CUDA_VISIBLE_DEVICES=2 python main.py --config config/SEUnet.yaml --world_size 1 --logname train_litsNoSpacing_seunet_bcel.log
nohup bash run.sh > train_litsNoSpacing_seunet_bcel.out 2>&1 &
tensorboard --logdir runs/lits_seg/seunet/train/2021-10-11-10-11-01/summarys
epoch9: train_dsc稳定上升至0.9472，loss降至0.666稳住了，tumor_dsc一直在0.6244
stop program

## train attunet
CUDA_VISIBLE_DEVICES=3 python main.py --config config/AttUnet.yaml --world_size 1 --logname train_litsNoSpacing_attunet_bcel.log
nohup bash run.sh > train_litsNoSpacing_attunet_bcel.out 2>&1 &
tensorboard --logdir runs/lits_seg/attunet/train/2021-10-11-10-14-09/summarys
epoch8: train_dsc稳定上升至0.94，loss降至0.6669稳住了，tumor_dsc一直在0.6244
loss稳定的不正常，tumor_dsc简直就是没有变化
epoch18: train_dsc稳定上升至0.9606，loss降至0.6663稳住了，tumor_dsc一直在0.6244
epoch 28: train_dsc稳定上升至0.9713，loss降至0.6659稳住了，tumor_dsc一直在0.6244
epoch 36: train_dsc稳定上升至0.9758，loss降至0.6657稳住了，tumor_dsc一直在0.6244

## train unet to debug tumor dsc
数据使用了 no-9
CUDA_VISIBLE_DEVICES=2 python main.py --config config/Unet.yaml --world_size 1 --logname train_lits_unet_bcel_debugTumorDsc.log
nohup bash run.sh > train_lits_unet_bcel_debugTumorDsc.out 2>&1 &
tensorboard --logdir runs/lits_seg/unet/train/2021-10-11-22-03-39/summarys
epoch13: train_dsc稳定上升至0.9458，loss降至0.6686稳住了，tumor_dsc一直在0.6421
loss稳定的不正常，tumor_dsc简直就是没有变化

# 1012
## train listLiver_attunet_bcel on cuda:1
使用 no-9 (with spacing)
CUDA_VISIBLE_DEVICES=1 python main.py --config config/AttUnet.yaml --world_size 1 --logname train_listLiver_attunet_bcel.log
nohup bash run.sh > train_listLiver_attunet_bcel.out 2>&1 &
tensorboard --logdir runs/lits_liver_seg/attunet/train/2021-10-12-10-16-13/summarys
10-12 20:21, 20 epoch, 可以继续train
达到对比tumor实验目标，stop training

## train attunet_listTumor_bcel on cuda:2
使用 no-9 (with spacing)
CUDA_VISIBLE_DEVICES=2 python main.py --config config/AttUnet.yaml --world_size 1 --logname train_listTumor_attunet_bcel.log
nohup bash run.sh > train_listTumor_attunet_bcel.out 2>&1 &
### val
tensorboard --logdir runs/lits_liver_seg/attunet/train/2021-10-12-10-27-35/summarys
10-12 20:29, 13 epoch, loss与dscs基本不变
达到tumor观察目标，stop training, try to test
### test
观察 dsc, sen 指标


# 1013
## train attunet_listTumor_dice on cuda:1

在liver上验证为什么loss在epoch 1就是0.666
## train 5000 slices show loss each iter
CUDA_VISIBLE_DEVICES=1 python main.py --config config/AttUnet.yaml --world_size 1 --logname train_listLiver_attunet_bcel_debugloss.log --debug
nohup bash run.sh > train_listLiver_attunet_bcel_debugloss.out 2>&1 &
### val
tensorboard --logdir runs/lits_liver_seg/attunet/train/2021-10-13-10-46-46/summarys
dsc反复折现，loss整体趋于下降，很快降至0.69

> 可能是网络末层 与 bcel中，同时使用了sigmoid，这样就使用了两次

## train seunet without sigmoid, 只看loss，metric无效
与上面一组实验进行对比
CUDA_VISIBLE_DEVICES=2 python main.py --config config/AttUnet.yaml --world_size 1 --logname train_listLiver_attunet_bcel_debugloss_delSigmoid.log --debug
nohup bash run.sh > train_listLiver_attunet_bcel_debugloss_delSigmoid.out 2>&1 &
### val
tensorboard --logdir runs/lits_liver_seg/attunet/train/2021-10-13-10-49-00/summarys
对比实验有效证明了是网络末层的sigmoid导致bcel loss一直在0.6之上徘徊

## train liver attunet baseline
在网络末层取消sigmoid函数，采用 attunet+lits+bcel
CUDA_VISIBLE_DEVICES=1 python main.py --config config/AttUnet.yaml --world_size 1 --logname train_litsLiver_attunetNoSigmoid_bcel.log
nohup bash run.sh > train_litsLiver_attunetNoSigmoid_bcel.out 2>&1 &
### val
tensorboard --logdir runs/lits_liver_seg/attunet/train/2021-10-13-16-21-24/summarys

## train tumor attunet baseline
在网络末层取消sigmoid函数，采用 attunet+tumor+bcel
CUDA_VISIBLE_DEVICES=2 python main.py --config config/AttUnet.yaml --world_size 1 --logname train_litsTumor_attunetNoSigmoid_bcel.log
nohup bash run.sh > train_litsTumor_attunetNoSigmoid_bcel.out 2>&1 &
### val
tensorboard --logdir runs/lits_liver_seg/attunet/train/2021-10-13-16-35-28/summarys

## train attunet + dice loss

探索为什么tumor dice一直在0.6244

肿瘤主要看 sensitivity ,  肿瘤像素中有多少是被找了出来

数据集的处理还可以 set HU window

# 1022
## 修改3d项目上的`LitsMyself.py`
CUDA_VISIBLE_DEVICES=2 python main.py --config config/Unet.yaml --world_size 1 --logname train_litsmyself_unet_bcel-debug_dataset.log

CUDA_VISIBLE_DEVICES=1 python main.py --config config/Unet.yaml --world_size 1 --logname train_litsmyself_unet_bcel.log
nohup bash run.sh > train_litsmyself_unet_bcel.out 2>&1 &

## train trasunet
CUDA_VISIBLE_DEVICES=3 python main.py --config config/TransUNet.yaml --world_size 1 --logname train_litsmyself_unet_bcel.log

# 1025
从csffm2分流至3090
使用cuda:2 来train transunet
CUDA_VISIBLE_DEVICES=2,3 python main.py --config config/TransUNet.yaml --world_size 2 --logname train_litsmyself_transunet_bcel.log
nohup bash run.sh > train_litsmyself_transunet_bcel.out 2>&1 &