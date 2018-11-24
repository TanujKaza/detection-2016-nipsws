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
import sys
from wgan import *


# Read number of epoch to be trained, to make checkpointing
parser = argparse.ArgumentParser(description='Epoch:')
parser.add_argument("-n", metavar='N', type=int, default=0)
parser.add_argument("-d", metavar='dataset_name', type=str, default='')
parser.add_argument("-g", metavar='use_gan_reward', type=bool, default=False)
args = parser.parse_args()
epochs_id = int(args.n)
dataset_name = str(args.d)
use_gan_reward = bool(args.g)

def dimg(img):
	plt.imshow(img)
	plt.show()


if __name__ == "__main__":

	######## PATHS definition ########

	path_train = "../data/quickdraw/" + dataset_name + "/obj-in-image/train/"
	path_object = "../data/quickdraw/" + dataset_name + "/r128/"

	path_model = "../models/quickdraw/" + dataset_name + "/" 
	# path of where to store visualizations of search sequences
	path_ouptut_folder = '../data/quickdraw/' + dataset_name + '/obj-in-image/test_output/'
	os.makedirs(path_ouptut_folder , exist_ok=True)

	######## PARAMETERS ########

	# Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
	scale_subregion = float(3)/4
	scale_mask = float(1)/(scale_subregion*4)
	# How many steps can run the agent until finding one object
	number_of_steps = 10
	epochs = 2000
	gamma = 0.90
	epsilon = 1.0
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

	weight_directory = path_model
	if not os.path.exists(weight_directory):
		os.makedirs(weight_directory)

	weight_files = os.listdir(weight_directory)
	epoch_nums = [-1]
	for file in weight_files:
		epoch_num = int(os.path.splitext(file)[0][7:])
		epoch_nums.append(epoch_num)
	current_epoch = max(epoch_nums)
	epochs_id = current_epoch + 1

	# epochs_id = 0

	if epochs_id == 0:
		q_net_weights = None
	else:
		q_net_weights = weight_directory +  '_epoch_' + str(current_epoch) + '.hdf5'
	model = build_q_network(q_net_weights)
	model.summary()

	wgan = WGAN( dataset_name )

	######## LOAD IMAGE NAMES ########

	img_names = np.array([load_images_names_in_dot_data_set(path_train)])
	obj_names = np.array([load_images_names_in_dot_data_set(path_object)])

	imgs , objs = get_all_train_images(img_names, path_train , path_object)

	for i in range(epochs_id, epochs_id + epochs):
		for j in range(np.size(img_names)):
			img = np.array(imgs[j])
			obj = np.array(objs[j])

			img_name = img_names[0][j]

			step = 0
			region_img = img
			offset = (0, 0)
			size_mask = (img.shape[0], img.shape[1])
			original_shape = size_mask


			if not use_gan_reward:
				new_iou = calculate_overlapping(region_img , obj)			
			else:
				new_iou = calculate_datsets_belonging(region_img , wgan)			

			iou = new_iou

			# follow_iou function calculates at each time step which is the groun truth object
			# that overlaps more with the visual region, so that we can calculate the rewards appropiately
			# init of the history vector that indicates past actions (6 actions * 4 steps in the memory)
			history_vector = np.zeros([24])
			# computation of the initial state instead of vgg_based i am just using 
			state = get_state(region_img , history_vector)
			# status indicates whether the agent is still alive and has not triggered the terminal action
			status = 1
			action = 0
			reward = 0
			cum_reward = 0
			while (status == 1) and (step < number_of_steps) :
				qval = model.predict(state, batch_size=1)
				# we force terminal action in case actual IoU is higher than 0.5, to train faster the agent
				if new_iou > 0.5:
					action = 6
				# epsilon-greedy policy
				elif random.random() < epsilon:
					action = np.random.randint(1, 7)
				else:
					action = (np.argmax(qval))+1
				# terminal action
				if action == 6:
					if not use_gan_reward:
						new_iou = calculate_overlapping(region_img , obj)			
					else:
						new_iou = calculate_datsets_belonging(region_img , wgan)
					
					reward = get_reward_trigger(new_iou)
					cum_reward += reward
				# movement action, we perform the crop of the corresponding subregion
				else:
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

					if not use_gan_reward:
						new_iou = calculate_overlapping(region_img , obj)			
					else:
						new_iou = calculate_datsets_belonging(region_img , wgan)

					reward = get_reward_movement(iou, new_iou)
					iou = new_iou
					cum_reward += reward
				
				history_vector = update_history_vector(history_vector, action)
				new_state = get_state(region_img, history_vector)
				# Experience replay storage
				if len(replay) < buffer_experience_replay:
					replay.append((state, action, reward, new_state))
				else:
					if h < (buffer_experience_replay-1):
						h += 1
					h_aux = int(h)
					replay[h_aux] = (state, action, reward, new_state)
					minibatch = random.sample(replay, batch_size)
					X_train0 = []
					X_train1 = []
					y_train = []
					# we pick from the replay memory a sampled minibatch and generate the training samples
					for memory in minibatch:
						old_state, action, reward, new_state = memory
						old_qval = model.predict(old_state, batch_size=1)
						newQ = model.predict(new_state, batch_size=1)
						maxQ = np.max(newQ)
						y = np.zeros([1, 6])
						y = old_qval.T
						if action != 6: #non-terminal state
							update = (reward + (gamma * maxQ))
						else: #terminal state
							update = reward
						y[action-1] = update #target output
						X_train0.append(old_state[0][0,:,:,:])
						X_train1.append(old_state[1][0,:])
						y_train.append(y)
					X_train0 = np.array(X_train0)
					X_train1 = np.array(X_train1)
					y_train = np.array(y_train)
					X_train0 = X_train0.astype("float32")
					X_train1 = X_train1.astype("float32")
					y_train = y_train.astype("float32")
					y_train = y_train[:, :, 0]
					hist = model.fit([X_train0 , X_train1], y_train, batch_size=batch_size, epochs=1, verbose=0)
					state = new_state
				if action == 6:
					status = 0
					
				step += 1
			print(i , j , step , action , new_iou , cum_reward )

			# train the wgan if enabled
			if use_gan_reward:
				wgan.train(enable_plot=True)

		
		out_img = cv2.resize(region_img , (64,64))
		inp_img = cv2.resize(img , (64,64))
		gt_img = cv2.resize(obj , (64,64))

		disp_img = np.concatenate([inp_img , gt_img , out_img ] ,axis=-1)

		cv2.imwrite(path_ouptut_folder + str(i) + '.bmp'  , disp_img)

		if epsilon > 0.1:
			epsilon -= 0.1
		os.makedirs(path_model , exist_ok=True)
		string = path_model  + '_epoch_' + str(i) + '.hdf5'
		model.save_weights(string, overwrite=True)

