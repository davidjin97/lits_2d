import os
import torch
from vit_seg_modeling import VisionTransformer as ViT_seg
from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

if __name__ == "__main__":
    vit_name = "R50-ViT-B_16"
    num_classes = 2
    img_size = 512
    vit_patches_size = 16
    n_skip = 3

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))

    gpu_ids = "3"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    input = torch.randn(1, 1, 512, 512).to(device) # BCHW 
    net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes).to(device)
    # net.load_from(weights=np.load(config_vit.pretrained_path))
    output = net(input)

    print("input.shape: ", input.shape)
    print("output.shape: ", output.shape)
    print(f"min: {output.min()}, max: {output.max()}")