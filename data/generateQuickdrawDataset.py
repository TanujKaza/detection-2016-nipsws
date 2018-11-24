from numpy import random
import gc
import numpy as np
import pdb
import cv2
import os
import sys
import matplotlib.pyplot as plt

dataset_name = sys.argv[1]

data_dir = './quickdraw/' + dataset_name + '/r128/'
save_dir = './quickdraw/' + dataset_name + '/obj-in-image/'
os.makedirs(save_dir + 'test/' , exist_ok=True)
os.makedirs(save_dir + 'train/' , exist_ok=True)

list_files = os.listdir(data_dir)
test_num = int(len(list_files) / 5)

mode = ''

for count , file in enumerate(list_files):
	if count < test_num:
		mode = 'test/'
	else:
		mode = 'train/'

	obj_img = cv2.imread(data_dir + file , 0)
	obj_img = cv2.resize(obj_img , (32,32))
	_,obj_img = cv2.threshold(obj_img,127,255,cv2.THRESH_BINARY)
	file , ext = os.path.splitext(file)
	for i in range(5):
		bkg_img = np.zeros((128,128))
		tx = random.randint(0,64)
		ty = random.randint(0,64)
		bkg_img[tx:tx+32,ty:ty+32] = obj_img
		cv2.imwrite(save_dir + mode + file + '_' + str(i) + ext , bkg_img)
		# pdb.set_trace()
