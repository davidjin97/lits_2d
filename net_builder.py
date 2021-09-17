# from networks.SeUnet import SeUnet
# from networks.Th_unet import ResidualUNet3D
# from networks.AMNet import AMEA_deepvision_res2block
from networks.Unet3D import Unet3D
from networks.UNet import UNet
from networks.ResUnet.ResUnet import ResUnet

import logging
logger = logging.getLogger('global')

SUPPORT_NETS = {
                'Unet3d': Unet3D,
                'UNet': UNet,
                'ResUnet': ResUnet,
                # 'seunet': SeUnet,
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
