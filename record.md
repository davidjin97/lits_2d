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