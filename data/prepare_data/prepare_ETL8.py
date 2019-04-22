import os
import numpy as np
import codecs
from scipy.misc import imread, imsave
import scipy
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import json
import collections
import random
# import matplotlib.pyplot as plt

DATA = 'D:/work/data/Extracted/etl_952_singlechar_size_64/'

def read_txt_to_label(fname):
    label = []
    with open(fname, 'r', encoding="utf-8") as f:
        f.readline()
        for line in f:
            label.append(line.split()[1])
    return label

def resize_image(img, size=(128,128)):
    # pad to square
    pad_size = int(abs(img.shape[0] - img.shape[1]) / 2)
    if img.shape[0] < img.shape[1]:
        pad_dims = ((pad_size, pad_size), (0, 0))
    else:
        pad_dims = ((0, 0), (pad_size, pad_size))
    img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
    # resize
    img = scipy.misc.imresize(img, size)
    assert img.shape == size
    return img

img_paths = []
for char_folder in os.listdir(os.path.join(DATA, '952_train')):
	writers = os.listdir(os.path.join(DATA, '952_train', char_folder))
	c = 0
	for writer in writers: 
		c += 1
		if 'ETL8B-w3' in writer: #select character 2
			img_paths.append(os.path.join(DATA, '952_train', char_folder, writer))
			# break
		# if  c == 10:
		# 	break
#print (img_paths)
print(len(img_paths))

# def resize_image(img, size=(128,128)):
#     # pad to square
#     pad_size = int(abs(img.shape[0] - img.shape[1]) / 2)
#     if img.shape[0] < img.shape[1]:
#         pad_dims = ((pad_size, pad_size), (0, 0))
#     else:
#         pad_dims = ((0, 0), (pad_size, pad_size))
#     img = np.lib.pad(img, pad_dims, mode='constant', constant_values=255)
#     # resize
#     img = scipy.misc.imresize(img, size)
#     assert img.shape == size
#     return img

def draw_single_char(ch, font, canvas_size=128, x_offset=0, y_offset=0):
    img = Image.new("L", (canvas_size, canvas_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img

if __name__ == "__main__":
	# python train.py --dataroot data/datasets/ken/etl_952_singlechar_size_64/ --name ETL_0.1_from_original_dense5 --gpu_ids 0 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_5blocks --no_dropout --input_nc 1 --output_nc 1 --batchSize 32
	source_path = 'D:/work/data/Extracted/fonts/simhei.ttf'
	DATASET_NAME = 'etl_952_singlechar_size_64_0.7'
	ratioA = 0.2
	ratioB = 0.2
	print('----------------')
	source_font = ImageFont.truetype(source_path, size=128)
	charlist = []
	bitmaplist = []
	sourcelist = []
	label = read_txt_to_label('D:/work/data/Extracted/etl_952_singlechar_size_64/952_labels.txt')
	datafolder = os.path.join('data/datasets', DATASET_NAME)
	if not os.path.exists(datafolder):
	    os.makedirs(datafolder)
	trainA_path = os.path.join(datafolder, 'trainA')
	trainB_path = os.path.join(datafolder, 'trainB')
	testA_path = os.path.join(datafolder, 'testA')
	testB_path = os.path.join(datafolder, 'testB')
	folders = [trainA_path, trainB_path, testA_path, testB_path]
	for folder in folders:
	    if not os.path.exists(folder):
	        os.mkdir(folder)
	for bitmap_path in img_paths:
	    bitmap = 255-imread(bitmap_path, mode='L') # background is white
	    bitmap = resize_image(bitmap, (64, 64))
	    bitmaplist.append(bitmap)
	    index = os.path.basename(os.path.dirname(bitmap_path))
	    ch = label[int(index)]
	    charlist.append(ch)
	    source_img = np.array(draw_single_char(ch, font=source_font))
	    source_img = resize_image(source_img, (64, 64))
	    sourcelist.append(source_img)
	arr = np.arange(len(charlist))
	np.random.shuffle(arr)
	ntrainA = np.floor(float(ratioA) * len(charlist))
	count = 0
	for x in np.arange(len(arr)):
		ch = charlist[arr[x]]
		print(ch,'    ', ord(ch))
		bitmap = bitmaplist[arr[x]]
		if x <= ntrainA:
			imsave(os.path.join(trainA_path, str(ord(ch)) + '.png'), bitmap)
		else:
			imsave(os.path.join(testA_path, str(ord(ch)) + '.png'), bitmap)
		count += 1
	np.random.shuffle(arr)
	ntrainB = np.floor(float(ratioB) * len(charlist))
	print('len :n', len(arr), ' ntrainA:', ntrainA,' ntrainB: ', ntrainB)
	count = 0
	for x in np.arange(len(arr)):
	    ch = charlist[arr[x]]
	    source_img = sourcelist[arr[x]]
	    if x <= ntrainB:
	        imsave(os.path.join(trainB_path, str(ord(ch)) + '.png'), source_img)
	    else:
	        imsave(os.path.join(testB_path, str(ord(ch)) + '.png'), source_img)
	    count += 1