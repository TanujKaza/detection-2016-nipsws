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
from features import get_image_descriptor_for_image, obtain_compiled_vgg_16, vgg_16, \
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


if __name__ == "__main__":

	######## PATHS definition ########

	path_train = "../data/dot_without_bg/train/"
	path_model = "..models/model_dot_without_bg/"
	# path of where to store visualizations of search sequences
	path_testing_folder = '../data/dot_without_bg/test_output/'
	# path of VGG16 weights
	# path_vgg_model = "../vgg16_weights.h5"

	######## PARAMETERS ########

	# Scale of subregion for the hierarchical regions (to deal with 2/4, 3/4)
	scale_subregion = float(3)/4
	scale_mask = float(1)/(scale_subregion*4)
	# 1 if you want to obtain visualizations of the search for objects
	bool_draw = 0
	# How many steps can run the agent until finding one object
	number_of_steps = 10
	# Boolean to indicate if you want to use the two databases, or just one
	two_databases = 0
	epochs = 5
	gamma = 0.90
	epsilon = 1
	batch_size = 100
	# Pointer to where to store the last experience in the experience replay buffer,
	# are trained at the same time
	h = 0
	# Each replay memory (one for each possible category) has a capacity of 100 experiences
	buffer_experience_replay = 1000
	# Init replay memories
	replay = []
	reward = 0

	######## model ########
	model_rl = obtain_compiled_model()
	model_rl.summary()

	# If you want to train it from first epoch, first option is selected. Otherwise,
	# when making checkpointing, weights of last stored weights are loaded for a particular class object

	model = get_q_network()
	model.summary()

	######## LOAD IMAGE NAMES ########

	img_names = np.array([load_images_names_in_dot_data_set(path_train)])
	imgs = get_all_dot_images(img_names, path_train)

	######## LOAD Ground Truth values ########
	gt_masks = np.load('../data/bb_info.npy')

	for i in range(epochs_id, epochs_id + epochs):
		for j in range(np.size(img_names)):
			masked = 0
			not_finished = 1
			img = np.array(imgs[j])
			img_name = img_names[0][j]
			img_index = int(os.path.splitext(img_name)[0])
			gt_mask = gt_masks[img_index]
			gt_mask_img = np.zeros([img.shape[0] , img.shape[1]])
			gt_mask_img[gt_mask[0]:gt_mask[2] , gt_mask[1]:gt_mask[3] ] = 1

			# pdb.set_trace()
			# array_classes_gt_objects = get_ids_objects_from_annotation(annotation)
			region_mask = np.ones([img.shape[0], img.shape[1]])
			step = 0
			# Init background
			background = Image.new('RGBA', (10000, 2500), (255, 255, 255, 255))
			draw = ImageDraw.Draw(background)
			# this matrix stores the IoU of each object of the ground-truth, just in case
			# the agent changes of observed object
			region_image = img
			offset = (0, 0)
			size_mask = (img.shape[0], img.shape[1])
			original_shape = size_mask
			old_region_mask = region_mask
			region_mask = np.ones([img.shape[0], img.shape[1]])
			new_iou = calculate_iou(gt_mask_img , region_mask)			
			iou = new_iou


			# If the ground truth object is already masked by other already found masks, do not
			# use it for training
				# We check if there are still obejcts to be found
			# follow_iou function calculates at each time step which is the groun truth object
			# that overlaps more with the visual region, so that we can calculate the rewards appropiately
			# init of the history vector that indicates past actions (6 actions * 4 steps in the memory)
			history_vector = np.zeros([24])
			# computation of the initial state
			state = get_state(region_image, history_vector, model_rl)
			# status indicates whether the agent is still alive and has not triggered the terminal action
			status = 1
			action = 0
			reward = 0
			if step > number_of_steps:
				background = draw_new_sequences(i, step, action, draw, region_image, background,
											path_testing_folder, iou, reward, gt_mask, region_mask, img_name,
											bool_draw)
				step += 1
			while (status == 1) & (step < number_of_steps) & not_finished:
				qval = model.predict(state.T, batch_size=1)
				background = draw_new_sequences(i, step, action, draw, region_image, background,
											path_testing_folder, iou, reward, gt_mask, region_mask,
											img_name,bool_draw)
				step += 1
				print(step)
				# we force terminal action in case actual IoU is higher than 0.5, to train faster the agent
				if (i < 100) & (new_iou > 0.5):
					action = 6
				# epsilon-greedy policy
				elif random.random() < epsilon:
					action = np.random.randint(1, 7)
				else:
					action = (np.argmax(qval))+1
				# terminal action
				if action == 6:
					gt_mask_img = np.zeros([img.shape[0] , img.shape[1]])
					gt_mask_img[gt_mask[0]:gt_mask[2] , gt_mask[1]:gt_mask[3] ] = 1
					new_iou = calculate_iou(gt_mask_img , region_mask)
					reward = get_reward_trigger(new_iou)
					background = draw_new_sequences(i, step, action, draw, region_image, background,
											path_testing_folder, iou, reward, gt_mask, region_mask,
											img_name,bool_draw)
					step += 1
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

					region_image = region_image[offset_aux[0]:offset_aux[0] + size_mask[0],
								   offset_aux[1]:offset_aux[1] + size_mask[1]]
					region_mask[offset[0]:offset[0] + size_mask[0], offset[1]:offset[1] + size_mask[1]] = 1
					gt_mask_img = np.zeros([img.shape[0] , img.shape[1]])
					gt_mask_img[gt_mask[0]:gt_mask[2] , gt_mask[1]:gt_mask[3] ] = 1
					new_iou = calculate_iou(gt_mask_img , region_mask)
					reward = get_reward_movement(iou, new_iou)
					iou = new_iou
				
				history_vector = update_history_vector(history_vector, action)
				new_state = get_state(region_image, history_vector, model_rl)
				# Experience replay storage
				if len(replay) < buffer_experience_replay:
					replay.append((state, action, reward, new_state))
				else:
					if h < (buffer_experience_replay-1):
						h += 1
					h_aux = h
					h_aux = int(h_aux)
					replay[h_aux] = (state, action, reward, new_state)
					minibatch = random.sample(replay, batch_size)
					X_train = []
					y_train = []
					# we pick from the replay memory a sampled minibatch and generate the training samples
					for memory in minibatch:
						old_state, action, reward, new_state = memory
						old_qval = model.predict(old_state.T, batch_size=1)
						newQ = model.predict(new_state.T, batch_size=1)
						maxQ = np.max(newQ)
						y = np.zeros([1, 6])
						y = old_qval
						y = y.T
						if action != 6: #non-terminal state
							update = (reward + (gamma * maxQ))
						else: #terminal state
							update = reward
						y[action-1] = update #target output
						X_train.append(old_state)
						y_train.append(y)
					X_train = np.array(X_train)
					y_train = np.array(y_train)
					X_train = X_train.astype("float32")
					y_train = y_train.astype("float32")
					X_train = X_train[:, :, 0]
					y_train = y_train[:, :, 0]
					hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
					state = new_state
				if action == 6:
					status = 0
					masked = 1
					# we mask object found with ground-truth so that agent learns faster
					gt_mask_img = np.zeros(img.shape)
					gt_mask_img[gt_mask[0]:gt_mask[2] , gt_mask[1]:gt_mask[3] ] = 1
					# NOTE I am not sure why they are doing this 
					# img = mask_image_with_mean_background(gt_mask_img, img)
				else:
					masked = 0
		if epsilon > 0.1:
			epsilon -= 0.1
		string = path_model + str(t) + '_epoch_' + str(i) + 'h5'
		string2 = path_model + str(t) + 'h5'
		model.save_weights(string, overwrite=True)
		model.save_weights(string2, overwrite=True)

