from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import _Merge
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys, os
import numpy as np
import pdb
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras.backend as K
import argparse
from functools import partial

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGAN():
    def __init__( self, dataset_name ):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = ( self.img_rows, self.img_cols, self.channels )
        self.latent_dim = 100

        # Load dataset
        self.DIR = '../data/quickdraw/'
        self.dataset_name = dataset_name
        train_path = os.path.join( self.DIR, dataset_name, 'object.npy' )
        self.train_data = np.load( train_path )

        g_optimizer = Adam( 0.0002, 0.5 )
        c_optimizer = SGD( 0.0002, 0.5 )

        self.n_critic = 5

        # Build the generator
        self.generator = self.build_generator()
        self.critic = self.build_discriminator()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------
        # Freeze generator's layers while training critic
        self.generator.trainable = False        


        # Image input (real sample)
        real_img = Input(shape=self.img_shape)
        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)        

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)        

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=c_optimizer,
                                        loss_weights=[1, 1, 10])        

        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=g_optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator( self ):

        model = Sequential()

        model.add( Dense( 128 * 16 * 16,
                          activation='relu',
                          input_dim=self.latent_dim ) )
        model.add( Reshape( ( 16, 16, 128 ) ) )
        model.add( UpSampling2D() )
        model.add( Conv2D( 128, kernel_size=3, padding='same' ) )
        model.add( BatchNormalization( momentum=0.8 ) )
        model.add( Activation( 'relu' ) )
        model.add( Conv2D( 128, kernel_size=3, padding='same' ) )
        model.add( BatchNormalization( momentum=0.8 ) )
        model.add( Activation( 'relu' ) )
        model.add( UpSampling2D() )
        model.add( Conv2D( 64, kernel_size=3, padding='same' ) )
        model.add( BatchNormalization( momentum=0.8 ) )
        model.add( Activation( 'relu' ) )
        model.add( Conv2D( 64, kernel_size=3, padding='same' ) )
        model.add( BatchNormalization( momentum=0.8 ) )
        model.add( Activation( 'relu' ) )
        if self.img_rows == 128:
            model.add( UpSampling2D() )
            model.add( Conv2D( 64, kernel_size=3, padding='same' ) )
            model.add( BatchNormalization( momentum=0.8 ) )
            model.add( Activation( 'relu' ) )

        model.add( Conv2D( self.channels, kernel_size=3, padding='same' ) )
        model.add( Activation( 'tanh' ) )

        model.summary()

        noise = Input( shape=( self.latent_dim, ) )
        img = model( noise )

        return Model( noise, img )

    def build_discriminator( self ):

        model = Sequential()

        model.add( Conv2D( 32,
                           kernel_size=3,
                           strides=2,
                           input_shape=self.img_shape,
                           padding='same' ) )
        model.add( LeakyReLU( alpha=0.2 ) )
        model.add( Dropout( 0.25 ) )
        model.add( Conv2D( 64, kernel_size=3, strides=2, padding='same' ) )
        model.add( ZeroPadding2D( padding=( ( 0, 1 ), ( 0,1 ) ) ) )
        model.add( BatchNormalization( momentum=0.8 ) )
        model.add( LeakyReLU( alpha=0.2 ) )
        model.add( Dropout( 0.25 ) )
        model.add( Conv2D( 128, kernel_size=3, strides=2, padding='same' ) )
        model.add( BatchNormalization( momentum=0.8 ) )
        model.add( LeakyReLU( alpha=0.2 ) )
        model.add( Dropout( 0.25 ) )
        model.add( Conv2D( 256, kernel_size=3, strides=1, padding='same' ) )
        model.add( BatchNormalization( momentum=0.8 ) )
        model.add( LeakyReLU( alpha=0.2 ) )
        model.add( Dropout( 0.25 ) )
        model.add( Flatten() )
        model.add( Dense( 1 ) )

        model.summary()

        img = Input( shape=self.img_shape )
        validity = model( img )

        return Model( img, validity )

    def write_log(self, callback, names, logs, batch_no ):
        for name, value in zip( names, logs ):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary( summary, batch_no )
            callback.writer.flush()

    def train( self,
               epochs=11,
               batch_size=32,
               sample_interval=10,
               save_interval=10,
               enable_plot=False ):
        if enable_plot:
            log_path = self.DIR + self.dataset_name + '/graphs/wgan'
            callback = TensorBoard( log_path )
            callback.set_model( self.generator_model )
            train_names = [ 'D_loss','G_loss', ]
        # Adversarial ground truths
        valid = -np.ones( ( batch_size, 1 ) )
        fake = np.ones( ( batch_size, 1 ) )
        dummy = np.zeros( ( batch_size, 1 ) )
        for epoch in range( epochs ):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random half of images
                idx = np.random.randint( 0, self.train_data.shape[ 0 ], batch_size )
                imgs = self.train_data[ idx ]

                # Sample noise and generate a batch of new images
                noise = np.random.normal( 0, 1, ( batch_size, self.latent_dim ) )
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch( noise, valid )

            # Plot the progress
            if enable_plot:
                self.write_log( callback,
                                train_names,
                                np.asarray( [ d_loss[ 0 ],
                                              g_loss ] ),
                                epoch )
            print ( '%d [D loss: %f] [G loss: %f]' % \
                    ( epoch, d_loss[ 0 ], g_loss ) )

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_imgs( epoch )
            if epoch % save_interval == 0:
                save_dir = os.path.join( self.DIR,
                                         self.dataset_name,
                                         'wgan_saved_weights',
                                         'background' )
                os.makedirs( save_dir, exist_ok=True )
                save_name = os.path.join( save_dir, 'g_' + str( epoch ) + '.hdf5' )
                self.generator.save_weights( save_name )

    def sample_imgs( self, epoch ):
        r, c = 5, 5
        noise = np.random.normal( 0, 1, ( r * c, self.latent_dim ) )
        gen_imgs = self.generator.predict( noise )

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots( r, c )
        cnt = 0
        for i in range( r ):
            for j in range( c ):
                axs[ i, j ].imshow( gen_imgs[ cnt, : , : , 0 ], cmap='gray' )
                axs[ i, j ].axis( 'off' )
                cnt += 1
        sample_dir = os.path.join( self.DIR,
                                   self.dataset_name,
                                   'wgan-output',
                                   'background' )
        os.makedirs( sample_dir, exist_ok=True )
        fig.savefig( os.path.join( sample_dir, str( epoch ) + '.png' ) )
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Train the background generator' )
    parser.add_argument( 'dataset_name', help='dataset name' )
    args = parser.parse_args()
    wgan = WGAN( args.dataset_name )
    wgan.train(enable_plot=True)
