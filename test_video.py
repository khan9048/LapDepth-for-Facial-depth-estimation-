import os
import glob
import argparse
import time
from PIL import Image
import numpy as np
import PIL
import cv2
import torch
import numpy as np
from model import LDRN
import glob
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import imageio

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

from utlis1111 import predict, display_images, to_multichannel, scale_up
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Laplacian Depth Residual Network training on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Directory setting

parser.add_argument('--model_dir', type=str, default=r'NYU_LDRN_ResNext101_epoch35_synthe_org/epoch_35_loss_3.2691_1.pkl')
parser.add_argument('--img_dir', type=str, default=None)
parser.add_argument('--img_folder_dir', type=str, default='my_examples/*.jpg')

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

assert (args.img_dir is not None) or (args.img_folder_dir is not None), "Expected name of input image file or folder"

if args.cuda and torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    cudnn.benchmark = True
    print('=> on CUDA')
else:
    print('=> on CPU')

if args.pretrained == 'KITTI':
    args.max_depth = 80.0
elif args.pretrained == 'NYU':
    args.max_depth = 50000.0

print('=> loading model..')
Model = LDRN(args)
if args.cuda and torch.cuda.is_available():
    Model = Model.cuda()
Model = torch.nn.DataParallel(Model)
assert (args.model_dir != ''), "Expected pretrained model directory"
Model.module.load_state_dict(torch.load(args.model_dir))
pre = Model.eval()


def get_img_arr(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (640, 480))
    x = np.clip(np.asarray(im, dtype=float) / 255, 0, 1)
    return x

video_name = args.img_folder_dir

cap = cv2.VideoCapture(video_name)
out_video_name = 'output.MOV'
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (1280, 480))


def display_single_image(output, inputs=None, is_colormap=True):
    import matplotlib.pyplot as plt

    plasma = plt.get_cmap('plasma')

    imgs = []

    imgs.append(inputs)

    ##rescale output
    out_min = np.min(output)
    out_max = np.max(output)
    output = output - out_min
    outputs = output/out_max

    if is_colormap:
        rescaled = outputs[:, :, 0]
        pred_x = plasma(rescaled)[:, :, :3]
        imgs.append(pred_x)

    img_set = np.hstack(imgs)

    return img_set

count = 0
ret = True
while ret:
    ret, image = cap.read()
    if ret is False:
        break
    img_arr = get_img_arr(image)
    count += 1
    output = scale_up(2, pre(Model, img_arr, batch_size=1))
    pred = output.reshape(output.shape[1], output.shape[2], 1)
    img_set = display_single_image(pred, img_arr)
    plt.figure(figsize=(20, 10))
    plt.imshow(img_set)
    filename = 'img_' + str(count).zfill(4) + '.png'
    plt.savefig(os.path.join('image_results', filename), bbox_inches='tight')


