#### In-place/on-the-fly data augmentation
import os
import argparse
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='In fly data augmentation')

    parser.add_argument('--dir', '-d', help='path to dataset directory')


    args = parser.parse_args()

    os.makedirs('data/new_augmented_dataset')
    filelist = []

    directory = args.dir
    for filename in os.listdir(directory):
    	if filename.endswith(".bmp"):
    		filelist.append(os.path.join(directory, filename))
    	else:
    		continue
    # print(filelist)

    for x in filelist:
    	img = load_img(x)
    	data = img_to_array(img)
    	samples = expand_dims(data, 0)
    	datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,zoom_range=[0.5,1.0])

    	a = 0
    	for i in datagen.flow(samples, batch_size=1, save_to_dir='data/new_augmented_dataset', save_prefix='aug', save_format='bmp'):
    		a=a+1
    		if a ==5:
    			break