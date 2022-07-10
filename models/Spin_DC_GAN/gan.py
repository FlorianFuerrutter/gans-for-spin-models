import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from random import random
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers, activations
from keras import backend as K

#--------------------------------------------------------------------
#Hardcode singleT eval


#-------------------dis
class PeriodicPadding2D(keras.layers.Layer):
    '''Pads a image with periodic conditions'''

    def __init__(self, padding=2, *args, **kwargs):
        super(PeriodicPadding2D, self).__init__(*args, **kwargs)
        self.padding = padding

    #def compute_output_shape(self, input_shape):
    #    shape = tf.shape(input_shape)
    #    assert shape.size == 3  

    #    if shape[1] is not None:
    #      length_h = shape[1] + 2 * self.padding         
    #    else:
    #      length_h = None

    #    if shape[1] is not None:
    #      length_w = shape[2] + 2 * self.padding         
    #    else:
    #      length_w = None

    #    return tuple([shape[0], length_h, length_w])

    def call(self, inputs):  
        x    = inputs
        size = self.padding

        #assumes channel last format!

        #pad cols
        x = tf.concat([x[:, :, -size:], x, x[:, :, 0:size]], axis=2) 

        #pad rows
        x = tf.concat([x[:, -size:, :], x, x[:, 0:size, :]], axis=1)

        return x

    def get_config(self):
        config = super(PeriodicPadding2D, self).get_config()
        config.update({"padding": self.padding})
        return config

def dec_layer(dec_input, filter_size, kernel_size, kernel_initializer, strides=2, drop_rate=0.0, padding='same', use_bn=False):   
    dec = dec_input
    dec = layers.Conv2D(filter_size, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer)(dec)

    if use_bn:
        dec = layers.BatchNormalization()(dec)

    dec = layers.LeakyReLU(alpha=0.2)(dec)

    if drop_rate > 0.0:
        dec = layers.Dropout(drop_rate)(dec)

    return dec

def create_discriminator(image_res):
    init = keras.initializers.GlorotUniform() #RandomNormal(stddev = 0.03)

    #Structure
    image_input = layers.Input(shape=image_res)

    #x = layers.GaussianNoise(0.01)(image_input)

    #Add periodic bounding conditions
    x = PeriodicPadding2D(padding=1)(image_input)

    #-----------Decoder
    drop_rate = 0.0
      
    x = dec_layer(x,   64//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init, padding='valid') #32x32  #64//2
    x = dec_layer(x,  128//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #16x16                   #128//2
    x = dec_layer(x,  128//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #8x8                     #128//2
    #x = dec_layer(x, 128, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #4x4

    #----------- Activation-layer
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation=activations.sigmoid)(x)

    #----------- Model
    d_model = keras.models.Model(inputs = image_input, outputs = output, name="discriminator")
    return d_model

#-------------------gen
def enc_layer(enc_input, filter_size, kernel_size, kernel_initializer, strides=2, drop_rate=0.0, padding='same', use_bn=False):   
    enc = enc_input
    enc = layers.Conv2DTranspose(filter_size, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer)(enc)

    if use_bn:
        enc = layers.BatchNormalization()(enc)

    enc = layers.LeakyReLU(alpha=0.2)(enc)

    if drop_rate > 0.0:
        enc = layers.Dropout(drop_rate)(enc)

    return enc

def create_generator(latent_dim):
    init = keras.initializers.GlorotUniform() #RandomNormal(stddev = 0.03)

    #Structure
    latent_input = layers.Input(shape=latent_dim)

    x = layers.Dense(8 * 8 * 128//2, use_bias=False)(latent_input)     
    x = layers.Reshape((8, 8, 128//2))(x)

    #----------Encoder
    drop_rate = 0.0

    x = enc_layer(x,  128//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #16x16   128 //2
    x = enc_layer(x,  192//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #32x32   192 //2
    x = enc_layer(x,  256//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #64x64   256 //2
    #x = enc_layer(x,  1024//4, kernel_size=(4,4), strides=(1,1), drop_rate=drop_rate, kernel_initializer=init)

    # 128, 192, 256 , epoch 102 looks very good!!!! , 7k and 64 batch, tain 1.5 1.25 (up to 132)
    # 192, 256, 320 , 20k and 64 batch, tain 1.5 1.25  half ok up to 114
    # 130, 192, 256 , 30k and 64 batch, tain 1.5 1.25, gets better,but bad, up to 27
    # 130, 192, 256 , 10k and 128 batch, tain 1.5 1.25 colabsd at 42 els seems ok

    # --------
    #gen: 130 192 256 // 2, batch=64, data=10k, tain 1.5 1.25
    #dis:  64 128 128 // 2

    #------- Activation-layer
    output = layers.Conv2D(1, kernel_size=(5,5), strides=1, padding='same', activation=activations.tanh)(x)

    #----------- Model
    g_model = keras.models.Model(inputs = latent_input, outputs=output, name="generator")
    return g_model


#--------------------------------------------------------------------

def plot_images(generated_images, images_count, epoch, plot_path=""):
    fig = plt.figure(figsize=(5, 5))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.05, hspace=0.05)

    res = int(np.sqrt(images_count))
    for i in range(images_count):
        plt.subplot(res, res, i+1)
        plt.axis('off')

        generated_image = generated_images[i].numpy()
        generated_image = np.where(generated_image < 0.5, 0.0, 1.0)
        plt.imshow(generated_image, vmin=0.0, vmax=1.0)  
        
    if plot_path == "":
        plt.savefig("img/generated_{epoch}.png".format(epoch=epoch), bbox_inches='tight')
    else:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plt.savefig(plot_path + "/generated_{epoch}.png".format(epoch=epoch), bbox_inches='tight')
    plt.close()

@tf.function
def sample_generator_input(batch_size, latent_dim,):
    return tf.random.normal(shape=(batch_size, latent_dim))
  
#--------------------------------------------------------------------

@tf.function
def wasserstein_loss(y_true, y_pred):
    #true label smooth
    label_smoothing = 0.05
    y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    #convert [0,1] to [-1,1]
    y_true = 2.0 * y_true - 1.0
    y_pred = 2.0 * y_pred - 1.0

    #calc wasserstein loss
    return -tf.reduce_mean(y_true * y_pred)

class gan(keras.Model):
    def __init__(self, latent_dim, image_size):
        super().__init__()

        self.generator = create_generator(latent_dim)
        self.discriminator = create_discriminator(image_size)

        #self.generator.summary()
        #self.discriminator.summary()

        #--------------------------------------------
        self.latent_dim      = latent_dim
        self.image_size      = image_size
        self.save_path = "./model-saves/gan_" 
        self.plot_path = ""

        #fixed loss metrics here
        self.d_loss_metric = keras.metrics.Mean(name="d_loss") 
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(gan, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        #set loss functions
        self.d_loss_fn     = d_loss_fn
        self.g_loss_fn     = g_loss_fn
        
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0] 
      
        #--------------------------------------------
        #train discriminator

        #latent and noise
        latent_vectors = sample_generator_input(batch_size, self.latent_dim)

        #set labels
        real_labels = tf.zeros((batch_size, 1)) 
        fake_labels = tf.ones((batch_size, 1))
        noisy_real_labels = real_labels + 0.05 * tf.random.uniform(tf.shape(real_labels)) 
        noisy_fake_labels = fake_labels + 0.05 * tf.random.uniform(tf.shape(fake_labels)) 

        #Record operations for automatic differentiation, all watched variables within scope
        with tf.GradientTape() as tape: 

            generated_images = self.generator(latent_vectors)

            fake_predictions = self.discriminator(generated_images) 
            real_predictions = self.discriminator(real_images) 

            fake_loss = self.d_loss_fn(noisy_fake_labels, fake_predictions) 
            real_loss = self.d_loss_fn(noisy_real_labels, real_predictions)
            d_loss = fake_loss + real_loss

            #--------------
            
        #derivative of d_loss with respect to trainable_weights
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
 
        #--------------------------------------------
        #train generator
      
        #latent and noise
        latent_vectors = sample_generator_input(batch_size, self.latent_dim)

        #Record operations for automatic differentiation, all watched variables within scope
        with tf.GradientTape() as tape: 

            predictions = self.discriminator(self.generator(latent_vectors)) 
            g_loss      = self.g_loss_fn(real_labels, predictions) 

            #--------------

        #derivative of g_loss with respect to trainable_weights
        grads = tape.gradient(g_loss, self.generator.trainable_weights) 
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights)) 
 
        #--------------------------------------------
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

    def save(self, epoch, only_weights=True):
        if only_weights:
            self.generator.save_weights(self.save_path + "generator_weights_{epoch}".format(epoch=epoch))
            self.discriminator.save_weights(self.save_path + "discriminator_weights_{epoch}".format(epoch=epoch))

        else:            
            self.generator.save(self.save_path + "generator_full_{epoch}".format(epoch=epoch))
            self.discriminator.save(self.save_path + "discriminator_full_{epoch}".format(epoch=epoch))

    def load(self, epoch, only_weights=True):
        if only_weights:
            self.generator.load_weights(self.save_path + "generator_weights_{epoch}".format(epoch=epoch))
            self.discriminator.load_weights(self.save_path + "discriminator_weights_{epoch}".format(epoch=epoch))

        else: 
            self.generator = keras.models.load_model(self.save_path + "generator_full_{epoch}".format(epoch=epoch))
            self.discriminator = keras.models.load_model(self.save_path + "discriminator_full_{epoch}".format(epoch=epoch))

    def plot_print_model_config(self):
        self.generator.summary()
        self.discriminator.summary()

        tf.keras.utils.plot_model(self.generator, "generator.png", show_shapes=True)
        tf.keras.utils.plot_model(self.discriminator, "discriminator.png", show_shapes=True)

class train_callback(keras.callbacks.Callback):
    def __init__(self, latent_dim, plot_period=1, save_period=5):
        super(train_callback, self).__init__()
        self.latent_dim  = latent_dim
        self.plot_period = plot_period
        self.save_period = save_period

    def on_epoch_end(self, epoch, logs=None):
        
        #--------------------------------------------
        #save weights
        if ((epoch % self.save_period) == 0) and (epoch > 1 * self.save_period):
            self.model.save(epoch, only_weights=True)

        #--------------------------------------------
        #plot images
        if (epoch % self.plot_period) == 0:
            images_count = 16

            latent_vectors = sample_generator_input(images_count, self.latent_dim)
            generated_images = self.model.generator(latent_vectors)
            generated_images = (generated_images + 1.0) / 2.0

            plot_images(generated_images, images_count, epoch, self.model.plot_path)