# Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network

by ##Shristi Kumari (2016B4A70574P), Akriti Garg(2016B4A70480P)

## Overview

This is implementation of "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network" a deep learning based Single-Image Super-Resolution (SISR) model. We named it **DCSCN**.

The model structure is like below. We use Deep CNN with Residual Net, Skip Connection and Network in Network. A combination of Deep CNNs and Skip connection layers is used as a feature extractor for image features on both local and global area. Parallelized 1x1 CNNs, like the one called Network in Network, is also used for image reconstruction.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/figure1.jpeg" width="800">


## Sample result

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/result.png" width="864">


Our model, DCSCN is much lighter than other Deep Learning based SISR models. Here is a comparison chart of performance vs computation complexity from our paper.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/compare.png" width="600">


## Requirements

python > 3.5

tensorflow > 1.0, scipy, numpy and pillow


## Result of PSNR


## Result of SSIM


## Instructions to Run

## Data Augmentation
Execute the following command on dataset of your own choice. Three data augmentation techniques have been implemented in data_aug_keras.py. The augemented dataset can be found in **data/new_augmented_dataset** folder.


```
# generated augmented dataset
python data-aug-keras.py --dir='path of dataset-of-your-own-choice'

```

Result of augmenting any one image as a sample

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/graph.png" width="100">


## How to train

You can train with any datasets. Put your image files as a training dataset into the directory under **data** directory, then specify with --dataset arg. There are some other hyper paramters to train, check [args.py](https://github.com/jiny2001/dcscn-super-resolution/blob/master/helper/args.py) to use other training parameters.

Each training and evaluation result will be added to **log.txt**.

```
# training with yang91 dataset
python train.py --dataset yang91

# training with larger filters and deeper layers
python train.py --dataset yang91 --filters 128 --layers 10

# after training has done, you can apply super resolution on your own image file. (put same args which you used on training)
python sr.py --file your_file.png --dataset yang91 --filters 128 --layers 10
```





## Visualization

During the training, tensorboard log is available. You can use "--save_weights True" to add histogram and stddev logging of each weights. Those are logged under **tf_log** directory.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/model.png" width="400">

Also we log average PSNR of traing and testing, and then generate csv and plot files under **graphs** directory. Please note training PSNR contains dropout factor so it will be less than test PSNR. This graph is from training our compact version of DCSCN.

<img src="https://raw.githubusercontent.com/jiny2001/dcscn-super-resolution/master/documents/graph.png" width="400">
