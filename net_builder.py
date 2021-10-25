# from networks.SeUnet import SeUnet
# from networks.Th_unet import ResidualUNet3D
# from networks.AMNet import AMEA_deepvision_res2block
from networks.Unet3D import Unet3D
from networks.UNet import UNet
from networks.ResUnet.ResUnet import ResUnet
from networks.AttentionUNet.AttUNet import AttU_Net
from networks.SEUnet import SEUnet
from networks.TransUNet.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.TransUNet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

import logging
logger = logging.getLogger('global')

SUPPORT_NETS = {
                'Unet3d': Unet3D,
                'UNet': UNet,
                'ResUnet': ResUnet,
                'AttUnet': AttU_Net,
                'SEUnet': SEUnet,
                'TransUNet': (ViT_seg, CONFIGS_ViT_seg)
                # '3dresidual_unet': ResidualUNet3D,
                # 'AMEA_deepvision': AMEA_deepvision_res2block,
                # 'ResUnet3D': ResUnet3D}
            }


def build_net(net_name):
    net = SUPPORT_NETS.get(net_name, None)
    if net is None:
        logger.error('Current supporting nets: %s , Unsupport net: %s', SUPPORT_NETS.keys(), net_name)
        raise 'Unsupport net: %s' % net_name
    return net
