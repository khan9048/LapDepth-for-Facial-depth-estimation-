#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 17:53:05 2017

TODO: Add extrinsic support??

@author: serkan
"""
import ply
import reconstruct
import argparse
import numpy as np
import cv2


#Program that converts depth image into 3d point cloud
#
def parse_args():
    parser = argparse.ArgumentParser(description='Convert depth image into 3d point cloud')
    parser.add_argument('image', type=str,  help='Source image', default='test2_new.jpg')
    parser.add_argument('depthmap', type=str, help='Depth map of source image', default='test2_new_depth.jpg')
    parser.add_argument('output', type=str,  help="Output ply file", default='f_new.ply')
    
    parser.add_argument('fx', type=float, help="fx", default='1148.93617021')
    parser.add_argument('fy', type=float, help="fy", default='1150.38461538')
    parser.add_argument('cx', type=float, help="cx", default='750')
    parser.add_argument('cy', type=float, help="cy", default='500')
    
    parser.add_argument('-m','--mesh', action='store_true', help='Reconstruct mesh')
    parser.add_argument('-s', '--step', type=int, help='Step size', default=1)
    parser.add_argument('-c', '--color', type=str, help='Color type', default='vertex' , choices=['vertex','texture'])
    args = parser.parse_args()

    return args


def load_image(image_path, name, color = True):
    try:
        if (color):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(image_path,cv2.IMREAD_ANYDEPTH)
    except Exception as e:
        #raise Exception("{} image is incorrect.".format(name)) from e
        raise Exception("{} image is incorrect.".format(name))
    return image, {}


args = parse_args()
image, image_exif = load_image(args.image, "Source")
depthmap, _       = load_image(args.depthmap, "Depth map", False)

image_height = image.shape[0]
image_width  = image.shape[1]

depthmap_height, depthmap_width = depthmap.shape[0], depthmap.shape[1]

fx, fy = args.fx, args.fy
cx, cy = args.cx, args.cy

print("Image size = {}x{}".format(image_width, image_height))
print("Depth map size = {}x{}".format(depthmap_width,depthmap_height))

if (depthmap_width != image_width or depthmap_height != image_height):
    print("Depthmap size does not match with image size. Scaling depthmap...")
    depthmap = cv2.resize(depthmap, (image_width, image_height))

camera_params = np.array([[fx,  0, cx],
                          [ 0, fy, cy],
                          [ 0,  0,  1]])

print("Camera intrinsic matrix:\n{}".format(camera_params))
print("Reconstructing...")
scene = reconstruct.reconstruct3d(image,depthmap,camera_params,step=args.step,mesh=args.mesh)

print("Reconstruction completed. Saving model...")
scene.save(args.output,color=args.color)
print("Done")

