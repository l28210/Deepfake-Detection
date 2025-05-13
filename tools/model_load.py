import sys
sys.path.append('/home/l/test_self/deepfake_detect/')

import argparse
import os
import torchvision.transforms as transforms

from model.replknet import *
from PIL import Image

def model_load(model_path, device):
    model = create_RepLKNet31B(drop_path_rate=0.3, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--LARGE_KERNEL_CONV_IMPL', type=str, default='lib/RepLKNet-pytorch/cutlass/examples/19_large_depthwise_conv2d_torch_extension/',
                        help='hight efficient implementaion of conv')
    config = vars(parser.parse_args())

    device = torch.device('cuda')

    os.environ['LARGE_KERNEL_CONV_IMPL'] = config['LARGE_KERNEL_CONV_IMPL']
    model = model_load('/home/l/test_self/deepfake_detect/logs_central/central_seed114514/19.pth', device)
    img_path = '/home/l/test_self/deepfake_detect/data/archive/test/fake/0A266M95TD_mask.png'
    img = Image.open(img_path).convert('RGB')
    rgb_tensor = transforms.ToTensor()(img)
    print(rgb_tensor.shape)
    x = torch.unsqueeze(rgb_tensor, 0).to(device)
    print(x.shape)
    y = model(x)
    print(y)