# Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network

## Shristi Kumari (2016B4A70574P), Akriti Garg(2016B4A70480P)

[Project Presentation](https://github.com/ShristiK/DCSCN-super-resolution-NNFL-Dessign-Project/blob/master/NNFL%20Design%20Project.pdf)

## Overview

This is implementation of **"Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"** a deep learning based Single-Image Super-Resolution (SISR) model. DCSCN model provides state-of-the-art performance with 10 times lower computation cost using parallelized 1x1 CNNs which not only reduces the dimensions of the previous layer for faster computation with less information loss, but also adds more nonlinearity to enhance the potential representation of the network. **Single Image Super-Resolution (SISR)** is used in many fields like security video surveillance and medical imaging, video playing, websites display.

## Requirements

1. python >=3.5

2. tensorflow 1.13.1

3. scipy 1.1.0

4. pillow

5. numpy

## Data
For training purpose, publicly available datasets are taken and the distribution is 91 images from Yang  and 200 images from the Berkeley Segmentation Dataset

## Model Structure

The DCSCN model consists of the following 2 parts : 

**1. Feature Extraction Network**

**2. Image Detail Reconstruction Network**

In the Feature Extraction Network part 7 sets of 3x3 CNN, bias and Parametric ReLU units are cascaded. Each output of the units is passed to the next unit and simultaneously skipped to the reconstruction network.
Parametric ReLu is used to solve “Dying ReLu “ problem as it prevents weights from learning a large negative bias term and leads to better performance.
The Image reconstruction network consists of 2 parallelized CNN blocks which are concatenated together
The first consists of 1x1 convolution layer with PRelu and the second consists of a 1x1 layer followed by a 3x3 layer with PRelu as Activation function. After this  a 1x1 CNN layer is added
1x1 CNNs are used to reduce the input dimension  before generating the high resolution  pixels.

Below is a figure showing DCSCN the model structure.

<img src = "https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/figure1.jpeg" width="800">

Our keras.py file creates the model using keras with structure as mentioned in the paper.

## Instructions to Run

## Data Augmentation
Execute the following command on dataset of your own choice. Three data augmentation techniques have been implemented in data_aug_keras.py. The augemented dataset can be found in **data/new_augmented_dataset** folder.


```
# generated augmented dataset
python data-aug-keras.py --dir='path of dataset-of-your-own-choice'

```


## How to train

You can train with any datasets. Put your image files as a training dataset into the directory under **data** directory, then specify with --dataset arg. 
Each training and evaluation result will be added to **log.txt**.

```
# training with yang91 dataset
python train.py --dataset yang91
```

## Evaluate Metrics
To calculate PSNR and SSIM metrics following command is run:
The evaluated metrics are for **DIV2K** dataset.

```
python evaluate.py --test_dataset all --dataset yang_bsd_8 --layers 10 --filters 196 --min_filters 48 --last_cnn_size 3

```

## Test on an Image

The model can be run on a single image and its result is generated in **Output folder** as a super resolution of the image.
The following command is to be run.

```
python sr.py --file your_file.png --dataset yang_bsd_4 --filters_decay_gamma 1.5 
```

## Results




## Visualization

During the training, tensorboard log is available. You can use "--save_weights True" to add histogram and stddev logging of each weights. Those are logged under **tf_log** directory.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/model.png" width="400">

Also we log average PSNR of traing and testing, and then generate csv and plot files under **graphs** directory. Please note training PSNR contains dropout factor so it will be less than test PSNR. This graph is from training our compact version of DCSCN.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/graph.png" width="400">
