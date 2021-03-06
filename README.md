# LapDepth-for-Facial-depth-estimation

This repository is a Pytorch implementation of the paper [**"Towards Monocular Facial Depth Estimation: Past, Present and Future "**]

## Setup & Requirements
To run this project, install it locally using pip install...:

```
$ pip install keras, Pillow, matplotlib, opencv-python, scikit-image, sklearn, pathlib, pandas, -U efficientnet,
$ pip install https://www.github.com/keras-team/keras-contrib.git, torch, torchvision
```

```
$ Python >= 3.6
Pytorch >= 1.6.0
Ubuntu 16.04
CUDA 9.2
cuDNN (if CUDA available)
```
## Pretrained model

download the pre-trained model and keep in the same directory:

https://nuigalwayie-my.sharepoint.com/:u:/r/personal/f_khan4_nuigalway_ie/Documents/PhD_projects/PhD_Papers/FACIAL_DEPTH_PAPER/epoch_25_loss_3.2482_1.pkl?csf=1&web=1&e=9LXlEG

## Prepare Dataset for training & Testing 

We prepared the dataset for training and testing<br/>
https://ieee-dataport.org/documents/c3i-synthetic-face-depth-dataset <br/>

## Virtual Human, Blender, Full Human Model, Avatar Dataset, 3D Data, 3D Full Body Models can be find here 
https://ieee-dataport.org/documents/c3i-synthetic-human-dataset <br/>

## Testing
First make sure that you have some images (RGB and depth)
```shell
$ Change the path in the demo.py
$ Name a folder that you want to save the predicted results (images)  
```
Once the preparation steps completed, you can test using following commands.
```
$ python demo.py
```
## Training
Once the dataset download process completed, please make sure unzip all the data into a new folder and follow the following steps:
```shell
$ Download the .csv or .txt files
$ Change the paths in the option.py  
```
Once the preparation steps completed, you can train using following commands.
```
$ python train.py 
```
## Evaluation
$ Change the paths in the eval_with_pngs.py  
$ Use the ground truth depth and predicted depth images for evaluation
```
Once the preparation steps completed, you can evaluate using following commands.
```
$ python eval_with_pngs.py 
```
## Point cloud generation
$ Go to the point cloud directory  
$ Use the images and demo.py for point cloud 
$ Change the paths in the demo.py if different 
$ python demo.py 
```
## Citation
If you find this work useful for your research, please consider citing our paper:
```
