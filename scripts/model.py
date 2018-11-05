import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
import os
import cv2
import numpy as np
import pdb
from keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers.pooling import MaxPooling2D

seed = 7
numpy.random.seed(seed)


def write_log(callback, names, logs, batch_no):
	for name, value in zip(names, logs):
		summary = tf.Summary()
		summary_value = summary.value.add()
		summary_value.simple_value = value
		summary_value.tag = name
		callback.writer.add_summary(summary, batch_no)
		callback.writer.flush()


class Model(object):
	def __init__(self):
		self.batch_size = 64
		self.img_rows = 128
		self.img_cols = 128
		self.channels = 1
		self.img_shape = (self.img_rows, self.img_cols, self.channels)
		optimizer = Adam(0.0001, 0.5)
		self.model = self.build_model()
		self.model.compile(loss='mean_squared_error',
								 optimizer=optimizer,
								 metrics=['mse'])
		self.image_test = []
		self.y_test = []

	def build_model(self, weights_path=None):
		model = Sequential()
		model.add(Conv2D(4, (3, 3), activation='relu' , input_shape=(self.img_rows, self.img_cols , self.channels) , padding='same') )
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(4, (3, 3), activation='relu'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(8,( 3, 3), activation='relu'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(8,( 3, 3), activation='relu'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(8,( 3, 3), activation='relu'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(16,( 3, 3), activation='relu'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(16,( 3, 3), activation='relu'))
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(16,( 3, 3), activation='relu'))
		model.add(MaxPooling2D((2, 2), strides=(2, 2)))

		model.add(Flatten())
		model.add(Dense(8))
		model.add(Dense(4))

		if weights_path:
			model.load_weights(weights_path)

		return model

	def train(self ,epochs):
		image_train = []
		y_train = []

		train_names = ['text_loss']
		log_path = './graphs/descriptor-model'
		callback = TensorBoard(log_path)
		callback.set_model(self.model)

		self.DIR = '../data/dot_without_bg/'
		directory = self.DIR + 'train'
		data = np.load('../data/bb_info.npy')
		for file_name in  os.listdir( directory ):
			if file_name.endswith( 'bmp' ):
				img = cv2.imread( directory + '/' + file_name, cv2.IMREAD_GRAYSCALE )
				image_train.append( img )
				image_number, file_extension = os.path.splitext(file_name) 
				y_train.append(data[int(image_number)])

		image_train = np.array( image_train )
		y_train = np.array( y_train )
				# Rescale -1 to 1
		image_train = ( image_train.astype( np.float32 ) - 127.5 ) / 127.5
		image_train = np.expand_dims( image_train, axis=3 )		

		y_train = np.expand_dims( y_train, axis=3 )
	
		def unison_shuffled_copies(a, b):
			assert len(a) == len(b)
			p = numpy.random.permutation(len(a))
			return a[p], b[p]

		y_train , image_train = unison_shuffled_copies(y_train , image_train)

		test_size = 100

		for file_name in  os.listdir( directory ):
			if file_name.endswith( 'bmp' ):
				img = cv2.imread( directory + '/' + file_name, cv2.IMREAD_GRAYSCALE )
				self.image_test.append( img )
				image_number, file_extension = os.path.splitext(file_name) 
				self.y_test.append(data[int(image_number)])

		self.image_test = np.array( self.image_test )
		self.y_test = np.array( self.y_test )
				# Rescale -1 to 1
		self.image_test = ( self.image_test.astype( np.float32 ) - 127.5 ) / 127.5
		self.image_test = np.expand_dims( self.image_test, axis=3 )		

		self.y_test = np.expand_dims( self.y_test, axis=3 )

		# pdb.set_trace()
		print("Training")
		for epoch in range(epochs):
			idx = np.random.randint( 0, image_train.shape[ 0 ], self.batch_size )
			imgs = image_train[ idx ]
			labels = y_train[idx]
			labels = tf.squeeze(labels, axis=2)
			loss = self.model.train_on_batch( imgs, labels )

			write_log(callback, train_names, np.asarray([loss[0]]), epoch)
			print ("%d [image: %s] " % \
			   (epoch, str(loss[0]) ))

			if epoch % 1000 == 0:
				self.save_model(epoch)

			if epoch % 100 == 0:
				self.check_model(epoch)


	def save_model(self , epoch):
		def save(model, model_name):
			os.makedirs('../models/decriptor_model' , exist_ok=True)
			model_path = "../models/decriptor_model/%s.json" % model_name
			weights_path = "../models/decriptor_model/%s_weights.hdf5" % model_name
			options = {"file_arch": model_path,
						"file_weight": weights_path}
			json_string = model.to_json()
			open(options['file_arch'], 'w').write(json_string)
			model.save_weights(options['file_weight'])

		save(self.model, "model" + str(epoch))

	def check_model(self , epoch):
		y_test = self.y_test
		image_test = self.image_test
		r, c = 1, 6
		idx = np.random.randint( 0, image_test.shape[ 0 ], c )

		imgs = image_test[idx]
		labels = y_test[idx]
		labels = np.squeeze(labels, axis=2)
		pred = self.model.predict(imgs)
		# pdb.set_trace()



if __name__ == '__main__':
	model = Model()
	model.train( epochs=20000 )


