import cv2, numpy as np
import time
import math as mth
from PIL import Image, ImageDraw, ImageFont
import scipy.io
from keras.models import Sequential
from keras import initializers
from keras.initializers import normal, identity
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD, Adam
import random
import argparse
from scipy import ndimage
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder
from features import get_image_descriptor_for_image, \
	get_conv_image_descriptor_for_image, calculate_all_initial_feature_maps
from image_helper import *
from metrics import *
from visualization import *
from reinforcement import *
import pdb
import matplotlib.pyplot as plt


# Read number of epoch to be trained, to make checkpointing
parser = argparse.ArgumentParser(description='Epoch:')
parser.add_argument("-n", metavar='N', type=int, default=0)
args = parser.parse_args()
epochs_id = int(args.n)

def dimg(img):
	plt.imshow(img)
	plt.show()


if __name__ == "__main__":

	######## PATHS definition ########

	path_test = "../data/dot_without_bg/test/"
	path_model = "../models/model_dot_without_bg/"
	# path of where to store visualizations of search sequences
	path_ouptut_folder = '../data/dot_without_bg/test_output1/'
	os.makedirs(path_ouptut_folder , exist_ok=True)

	# weight_descriptor_model = "../models/decriptor_model/model1000_weights.hdf5"

	######## PARAMETERS ########

	# Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
	scale_subregion = float(3)/4
	scale_mask = float(1)/(scale_subregion*4)
	# 1 if you want to obtain visualizations of the search for objects
	bool_draw = 1
	# How many steps can run the agent until finding one object
	number_of_steps = 10
	epochs = 2000
	gamma = 0.90
	epsilon = 0.0
	batch_size = 10
	test_size = 1
	# Pointer to where to store the last experience in the experience replay buffer,
	# are trained at the same time
	h = 0
	# Each replay memory (one for each possible category) has a capacity of 100 experiences
	buffer_experience_replay = 10
	# Init replay memories
	replay = []
	reward = 0

	######## model ########
	# model_rl = obtain_compiled_model(weight_descriptor_model)
	# model_rl.summary()

	# If you want to train it from first epoch, first option is selected. Otherwise,
	# when making checkpointing, weights of last stored weights are loaded for a particular class object

	weight_directory = '../models/model_dot_without_bg/'
	weight_files = os.listdir(weight_directory)
	epoch_nums = [-1]
	for file in weight_files:
		epoch_num = int(os.path.splitext(file)[0][7:])
		epoch_nums.append(epoch_num)
	current_epoch = max(epoch_nums)
	epochs_id = current_epoch + 1

	if epochs_id == 0:
		q_net_weights = None
	else:
		q_net_weights = weight_directory +  '_epoch_' + str(current_epoch) + '.hdf5'
	model = build_q_network(q_net_weights)
	model.summary()

	######## LOAD IMAGE NAMES ########

	img_names = np.array([load_images_names_in_dot_data_set(path_test)])
	imgs = get_all_dot_images(img_names, path_test)

	######## LOAD Ground Truth values ########
	gt_masks = np.load('../data/bb_info.npy')

	for j in range(np.size(img_names)):
		img = np.array(imgs[j])
		img_name = img_names[0][j]
		img_index = int(os.path.splitext(img_name)[0])
		gt_mask = gt_masks[img_index]
		gt_mask_img = np.zeros([img.shape[0] , img.shape[1]])
		gt_mask_img[gt_mask[1]:gt_mask[3] , gt_mask[0]:gt_mask[2] ] = 1

		region_mask = np.ones([img.shape[0], img.shape[1]])
		step = 0
		region_img = img
		offset = (0, 0)
		size_mask = (img.shape[0], img.shape[1])
		original_shape = size_mask
		old_region_mask = region_mask
		region_mask = np.ones([img.shape[0], img.shape[1]])
		new_iou = calculate_iou(gt_mask_img , region_mask)			
		iou = new_iou

		history_vector = np.zeros([24])
		state = get_state(region_img , history_vector)
		# status indicates whether the agent is still alive and has not triggered the terminal action
		status = 1
		action = 0
		reward = 0
		cum_reward = 0
		while (status == 1) and (step < number_of_steps) :
			qval = model.predict(state, batch_size=1)
			step += 1
			if new_iou > 0.5:
				action = 6
			# epsilon-greedy policy
			elif random.random() < epsilon:
				action = np.random.randint(1, 7)
			else:
				action = (np.argmax(qval))+1
			# terminal action
			if action == 6:
				new_iou = calculate_iou(gt_mask_img , region_mask)
				reward = get_reward_trigger(new_iou)
				cum_reward += reward
				step += 1
				status = 0
			# movement action, we perform the crop of the corresponding subregion
			else:
				region_mask = np.zeros(original_shape)
				size_mask = (size_mask[0] * scale_subregion, size_mask[1] * scale_subregion)
				if action == 1:
					offset_aux = (0, 0)
				elif action == 2:
					offset_aux = (0, size_mask[1] * scale_mask)
					offset = (offset[0], offset[1] + size_mask[1] * scale_mask)
				elif action == 3:
					offset_aux = (size_mask[0] * scale_mask, 0)
					offset = (offset[0] + size_mask[0] * scale_mask, offset[1])
				elif action == 4:
					offset_aux = (size_mask[0] * scale_mask, 
								  size_mask[1] * scale_mask)
					offset = (offset[0] + size_mask[0] * scale_mask,
							  offset[1] + size_mask[1] * scale_mask)
				elif action == 5:
					offset_aux = (size_mask[0] * scale_mask / 2,
								  size_mask[0] * scale_mask / 2)
					offset = (offset[0] + size_mask[0] * scale_mask / 2,
							  offset[1] + size_mask[0] * scale_mask / 2)

				offset = [int(x) for x in list(offset)]
				offset_aux = [int(x) for x in list(offset_aux)]
				size_mask = [int(x) for x in list(size_mask)]

				region_img = region_img[offset_aux[0]:offset_aux[0] + size_mask[0],
							   offset_aux[1]:offset_aux[1] + size_mask[1]]
				region_mask[offset[0]:offset[0] + size_mask[0], offset[1]:offset[1] + size_mask[1]] = 1
				new_iou = calculate_iou(gt_mask_img , region_mask)
				reward = get_reward_movement(iou, new_iou)
				iou = new_iou
				cum_reward += reward
			
			history_vector = update_history_vector(history_vector, action)
			new_state = get_state(region_img, history_vector)
			step += 1
		print(j , step , action , new_iou , cum_reward )
		
		out_img = cv2.resize(region_img , (64,64))
		inp_img = cv2.resize(img , (64,64))
		inp_mask_img = cv2.resize(gt_mask_img*255 , (64,64) )
		out_mask_img = cv2.resize(region_mask*255 , (64,64) )

		disp_img = np.concatenate([inp_img , inp_mask_img , out_img , out_mask_img] ,axis=-1)

		cv2.imwrite(path_ouptut_folder + str(j) + '.bmp'  , disp_img)
