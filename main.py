import os
import argparse
import datetime
import random
import logging
import shutil
import yaml
from easydict import EasyDict

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from utils.utils import format_cfg
from utils.setup_logging import init_logging
from trainer_lit import SegTrainer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def start(rank, world_size, args):
    setup(rank, world_size)
    init_logging('global', logging.INFO, args.logname, rank)
    logger = logging.getLogger('global')

    torch.cuda.is_available()
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    config_file = args.config
    logger.info('cfg_file: {}'.format(config_file))
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = EasyDict(config)
    
    #设置随机种子
    config.manualSeed = random.randint(1, 10000)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    torch.cuda.manual_seed_all(config.manualSeed)

    #是否需要转模型
    config['evaluate'] = args.evaluate
    config['tocaffe'] = args.tocaffe
    config['toonnx'] = args.toonnx
    config.rank = rank
    config.world_size = world_size

    config['stride'] = 64
    config.debug = args.debug
    logger.info("Running with config:\n{}".format(format_cfg(config)))

    trainer = SegTrainer(config)
    if not config.evaluate:
        train_outdir = config['train_output_directory']
        os.makedirs(train_outdir, exist_ok=True)
        logger.info('This is for traininig!')
        trainer.train()
    else:
        trainer.test()
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='implementation of Brain Seg')
    parser.add_argument('--config', dest='config', required=True)
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
    parser.add_argument('--world_size',dest='world_size', type=int, default=1) # 单线程
    parser.add_argument('--logname',dest='logname',default='train_jzw.log')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    assert (os.path.exists(args.config))
    mp.spawn(start, args=(args.world_size, args), nprocs=args.world_size)