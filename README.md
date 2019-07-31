# 3D Point Cloud Denoising via Deep Neural Network based Local Surface Estimation

by Chaojing Duan, Siheng Chen and Jelena Kovacevic

The code is written by Muqiao Yang.

Introduction
This repository is for our ICASSP 2019 paper '3D Point Cloud Denoising via Deep Neural Network based Local Surface Estimation'. The code is modified from PointNet and FoldingNet.

Installation
This code has been tested with Pytorch 0.4, Python 3.6 and Ubuntu 14+ . We will upload a new version based on Pytorch 1.0 since it is more stable and convenient.

Dataset
ShapeNet meshes:
https://www.shapenet.org/

ModelNet meshes:
http://modelnet.cs.princeton.edu/

You can also train with the dataset in PUNet, which concentrates more on the local geometries:
https://github.com/yulequan/PU-Net

Please cite their papers if you use their dataset to train/test.

Experiments
### With feature transform
```
python3.6 utils/train.py --path <dataset path> --filename <file name> --nepoch <num of epochs> --feature_transform
```

### Without feature transform
```
python3.6 utils/train.py --path <dataset path> --filename <file name> --nepoch <num of epochs>
```
