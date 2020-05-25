# Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network

## Shristi Kumari (2016B4A70574P), Akriti Garg(2016B4A70480P)

[Project Presentation] (https://github.com/ShristiK/DCSCN-super-resolution-NNFL-Dessign-Project/blob/master/NNFL%20Design%20Project.pdf)

## Overview

This is implementation of "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network" a deep learning based Single-Image Super-Resolution (SISR) model. We named it **DCSCN**.

The model structure is like below. The paper uses Deep CNN with Residual Net, Skip Connection and Network in Network. A combination of Deep CNNs and Skip connection layers is used as a feature extractor for image features on both local and global area. Parallelized 1x1 CNNs, like the one called Network in Network, is also used for image reconstruction.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/figure1.jpeg" width="800">



## Requirements

python >=3.5
tensorflow 1.13.1
scipy 1.1.0
pillow
numpy





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
