
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import os
import argparse
import numpy as np
import torch
import torch
from model import LDRN
import glob
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import argparse
import os
from utils import *
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
from trainer import validate
from model import *

parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting

parser.add_argument('--model_dir', type=str, default=r'NYU_LDRN_ResNext101_epoch35_synthe_org/epoch_25_loss_3.2482_1.pkl')
parser.add_argument('--img_dir', type=str, default=None)

# Dataloader setting
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')

# Model setting
parser.add_argument('--encoder', type=str, default="ResNext101")
parser.add_argument('--pretrained', type=str, default="NYU")
parser.add_argument('--norm', type=str, default="BN")
parser.add_argument('--n_Group', type=int, default=32)
parser.add_argument('--reduction', type=int, default=16)
parser.add_argument('--act', type=str, default="ReLU")
parser.add_argument('--max_depth', default=80.0, type=float, metavar='MaxVal', help='max value of depth')
parser.add_argument('--lv6', action='store_true', help='use lv6 Laplacian decoder')

# GPU setting
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu_num', type=str, default="0", help='force available gpu index')
parser.add_argument('--rank', type=int, help='node rank for distributed training', default=0)

args = parser.parse_args()

# assert (args.img_dir is not None) or (args.img_folder_dir is not None), "Expected name of input image file or folder"

if args.cuda and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    cudnn.benchmark = True
    print('=> on CUDA')
else:
    print('=> on CPU')

if args.pretrained == 'KITTI':
    args.max_depth = 80.0
elif args.pretrained == 'NYU':
    args.max_depth = 1000.0


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
		                                transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225) )])
        img = transform(img)
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img


if __name__ == '__main__':


    # create depth model

    print('=> loading model..')
    depth_model = LDRN(args)
    if args.cuda and torch.cuda.is_available():
        depth_model = depth_model.cuda()
    depth_model = torch.nn.DataParallel(depth_model)
    assert (args.model_dir != ''), "Expected pretrained model directory"
    depth_model.module.load_state_dict(torch.load(args.model_dir))
    depth_model.eval()


    image_dir = 'test_images/'
    imgs_list = os.listdir(image_dir)
    imgs_list.sort()
    imgs_path = [os.path.join(image_dir, i) for i in imgs_list if i != 'outputs']
    image_dir_out = image_dir + '/outputs'
    os.makedirs(image_dir_out, exist_ok=True)

    for i, v in enumerate(imgs_path):
        print('processing (%04d)-th image... %s' % (i, v))
        rgb = cv2.imread(v)
        rgb_c = rgb[:, :, ::-1].copy()
        gt_depth = None
        A_resize = cv2.resize(rgb_c, (448, 448))
        rgb_half = cv2.resize(rgb, (rgb.shape[1]//2, rgb.shape[0]//2), interpolation=cv2.INTER_LINEAR)


        img_torch = scale_torch(A_resize)[None, :, :, :]
        pred_depth = depth_model.inference(img_torch).cpu().numpy().squeeze()

        pred_depth_ori = cv2.resize(pred_depth, (rgb.shape[1], rgb.shape[0]))

        # if GT depth is available, uncomment the following part to recover the metric depth
        #pred_depth_metric = recover_metric_depth(pred_depth_ori, gt_depth)

        img_name = v.split('/')[-1]
        cv2.imwrite(os.path.join(image_dir_out, img_name), rgb)
        # save depth
        plt.imsave(os.path.join(image_dir_out, img_name[:-4]+'-depth.png'), pred_depth_ori, cmap='rainbow')
        cv2.imwrite(os.path.join(image_dir_out, img_name[:-4]+'-depth_raw.png'), (pred_depth_ori/pred_depth_ori.max() * 60000).astype(np.uint16))
