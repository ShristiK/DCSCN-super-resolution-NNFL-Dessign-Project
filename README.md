# Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network

## Shristi Kumari (2016B4A70574P), Akriti Garg(2016B4A70480P)

[Project Presentation]()

## Overview

This is an implementation of "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network" a deep learning based Single-Image Super-Resolution (SISR) model. A highly efficient and faster Single Image Super-Resolution (SISR) model with Deep Convolutional neural networks (Deep CNN) is proposed in the paper which achieves state-of-the-art performance at 10 times lower computation cost. Single Image Super-Resolution (SISR) is used in many fields like security video surveillance and medical imaging, video playing, websites display.

The current Single Image Super Resolution models involve large computations and are not suitable for network edge devices like mobile, tablet and IoT devices.DCSCN model provides state-of-the-art performance with 10 times lower computation cost using parallelized 1x1 CNNs which not only reduces the dimensions of the previous layer for faster computation with less information loss, but also adds more nonlinearity to enhance the potential representation of the network

## Requirements and Environment setup

python > 3.5

tensorflow > 1.0, scipy, numpy and pillow

## Data Description

For training purpose, publicly available datasets are taken and the distribution is 91 images from Yang and 200 images from the Berkeley Segmentation Dataset. Data augmentation is performed to obtain different sets of data like SET 5, 6 etc. In the training phase, SET 5 dataset is used to evaluate performance and check if the model will overfit or not. Each training image is split into 32 by 32 patches with stride 16 and 64 patches are used as a mini-batch.For testing, SET 5 , SET 14 , and BSDS100 have been used. 

Below is a sample data image.

<img src="">

## Data augmentation
Data augmentation is achieved by 3 different techniques invloving rotating, flipping and zooming an image. Below images are obtained after augmenting the given sample image using flip, zoom and rotate technique.

<img src = "">

## Model Overview

The DCSCN model consists of the following 2 parts : 

**1. Feature Extraction Network**

**2. Image Detail Reconstruction Network**

The Feature Extraction Network part consists of 7 sets of 3x3 CNN, bias and Parametric ReLU units. Each output of the units is passed to the next unit and simultaneously skipped to the reconstruction network. Parametric ReLu is used to solve “Dying ReLu “ problem as it prevents weights from learning a large negative bias term and leads to better performance. The Image reconstruction network consists of 2 parallelized CNN blocks which are concatenated together. The first consists of 1x1 convolution layer with PRelu and the second consists of a 1x1 layer followed by a 3x3 layer with PRelu as Activation function. After this  a 1x1 CNN layer is added. 1x1 CNNs are used to reduce the input dimension  before generating the high resolution  pixels.

The image below shows the DCSCN model structure.

<img src="">

## Instructions to Run

### How to train

### Apply to your own image


## Sample result

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/result.png" width="864">


Our model, DCSCN is much lighter than other Deep Learning based SISR models. Here is a comparison chart of performance vs computation complexity from our paper.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/compare.png" width="600">

### Result of PSNR


### Result of SSIM


