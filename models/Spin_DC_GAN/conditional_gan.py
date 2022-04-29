import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'font.size': 13,
    'mathtext.fontset': 'cm',
    'font.family': 'STIXGeneral',
    'axes.unicode_minus': True})

import numpy as np
import tensorflow as tf
from tensorflow import keras
from random import random
import os

import generator
import discriminator
from keras import backend as K

#--------------------------------------------------------------------

def plot_images(generated_images, labels, images_count, epoch, plot_path=""):
    fig = plt.figure(figsize=(5, 5))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.05, hspace=0.05)

    res = int(np.sqrt(images_count))
    for i in range(images_count):
        plt.subplot(res, res, i+1)
        plt.axis('off')

        generated_image = generated_images[i].numpy()
        generated_image = np.where(generated_image < 0.5, 0.0, 1.0)
        plt.imshow(generated_image, vmin=0.0, vmax=1.0)  

        args = dict(horizontalalignment='left',verticalalignment='top',transform = plt.gca().transAxes, color="black",bbox=dict(facecolor='white', alpha=1, boxstyle="round", pad=0.1))
        plt.text(0.05, 0.95, r"%.2f" % labels[i, 0], args)

    if plot_path == "":
        plt.savefig("img/generated_{epoch}.png".format(epoch=epoch), bbox_inches='tight')
    else:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plt.savefig(plot_path + "/generated_{epoch}.png".format(epoch=epoch), bbox_inches='tight')
    plt.close()

@tf.function(jit_compile=True)
def sample_generator_input(batch_size, latent_dim,):
    return tf.random.normal(shape=(batch_size, latent_dim))
  
#--------------------------------------------------------------------

@tf.function(jit_compile=True)
def wasserstein_loss(y_true, y_pred):
    #true label smooth
    label_smoothing = 0.05
    y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

    #convert [0,1] to [-1,1]
    y_true = 2.0 * y_true - 1.0
    y_pred = 2.0 * y_pred - 1.0

    #calc wasserstein loss
    return -tf.reduce_mean(y_true * y_pred)

def gradient_penalty(samples, output, weight):   
    # Penalize the gradient norm
    # r1 = (weight / 2) * E( ||grad||^2 )

    gradients        = tf.gradients(output, samples)[0]
    gradients_sqr    = tf.square(gradients)
    gradient_penalty = tf.reduce_sum(gradients_sqr, axis=[1, 2, 3])

    return tf.reduce_mean(gradient_penalty) * weight

class conditional_gan(keras.Model):
    def __init__(self, latent_dim, conditional_dim, image_size):
        super().__init__()

        #---------------------------------------------
        self.use_aux = True

        self.generator = generator.create_generator(latent_dim + conditional_dim, 1)       
        self.discriminator = discriminator.create_discriminator(image_size, 1, self.use_aux)

        self.generator.summary()
        self.discriminator.summary()

        #--------------------------------------------
        self.latent_dim      = latent_dim
        self.conditional_dim = conditional_dim
        #self.dis_input_shape = (image_size[0], image_size[1], image_size[2] + cond_channels)

        self.save_path = "./model-saves/gan_" 
        self.plot_path = ""

        #fixed loss metrics here
        self.d_loss_metric = keras.metrics.Mean(name="d_loss") 
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(conditional_gan, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        #set loss functions
        self.d_loss_fn  = d_loss_fn
        self.g_loss_fn  = g_loss_fn
        self.a_loss_fn  = keras.losses.MeanSquaredError()
     
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    @tf.function(jit_compile=True)
    def train_step(self, data):
        #------#unpack train data
        real_images, conditional_labels = data

        shape = tf.shape(real_images)
        batch_size = shape[0] 
        h          = shape[1]
        w          = shape[2]
        c          = tf.shape(conditional_labels)[-1]

        conditional_latent = tf.repeat(conditional_labels, repeats=[self.conditional_dim])
        conditional_latent = tf.reshape(conditional_latent, (-1, self.conditional_dim))

        conditional_channel = conditional_labels[:, :, None, None]
        conditional_channel = tf.repeat(conditional_channel, repeats=[h * w])
        conditional_channel = tf.reshape(conditional_channel, (-1, h, w, c))
    
        real_conditional_images = tf.concat([real_images, conditional_channel], -1)
        
        #conditional_labels_f32 = tf.dtypes.cast(conditional_labels, tf.float32)

        #--------------------------------------------
        #train discriminator

        #latent and noise     
        random_vectors = sample_generator_input(batch_size, self.latent_dim)
        latent_vectors = tf.concat([random_vectors, conditional_latent], axis=1)

        #set labels
        real_labels = tf.zeros((batch_size, 1)) 
        fake_labels = tf.ones((batch_size, 1))
        noisy_real_labels = real_labels + 0.05 * tf.random.uniform(tf.shape(real_labels)) 
        noisy_fake_labels = fake_labels + 0.05 * tf.random.uniform(tf.shape(fake_labels)) 

        #Record operations for automatic differentiation, all watched variables within scope
        with tf.GradientTape() as tape: 

            generated_images = self.generator(latent_vectors)
            generated_conditional_images = tf.concat([generated_images, conditional_channel], -1)

            fake_predictions = self.discriminator(generated_conditional_images) 
            real_predictions = self.discriminator(real_conditional_images) 

            #fake_loss = self.d_loss_fn(noisy_fake_labels, fake_predictions) 
            #real_loss = self.d_loss_fn(noisy_real_labels, real_predictions) 
            fake_loss = self.d_loss_fn(noisy_fake_labels, fake_predictions[0])
            real_loss = self.d_loss_fn(noisy_real_labels, real_predictions[0]) + 5.0 * self.a_loss_fn(conditional_labels, real_predictions[1])
          
            d_loss = fake_loss + real_loss + gradient_penalty(real_conditional_images, real_predictions, 5.0)

            #--------------
            
        #derivative of d_loss with respect to trainable_weights
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
 
        #--------------------------------------------
        #train generator                
        random_conditional = conditional_labels + tf.random.normal(tf.shape(conditional_labels), stddev=0.03)

        random_conditional_latent = tf.repeat(random_conditional, repeats=[self.conditional_dim])
        random_conditional_latent = tf.reshape(random_conditional_latent, (-1, self.conditional_dim))

        random_conditional_channel = random_conditional[:, :, None, None]
        random_conditional_channel = tf.repeat(random_conditional_channel, repeats=[h * w])
        random_conditional_channel = tf.reshape(random_conditional_channel, (-1, h, w, c))

        #latent and noise 
        random_vectors = sample_generator_input(batch_size, self.latent_dim)
        latent_vectors = tf.concat([random_vectors, random_conditional_latent], axis=1)

        #Record operations for automatic differentiation, all watched variables within scope
        with tf.GradientTape() as tape: 

            generated_images = self.generator(latent_vectors)
            generated_conditional_images = tf.concat([generated_images, random_conditional_channel], -1)

            predictions = self.discriminator(generated_conditional_images) 

            #g_loss = self.g_loss_fn(real_labels, predictions) 
            g_loss = self.g_loss_fn(real_labels, predictions[0]) + 5.0 * self.a_loss_fn(random_conditional, predictions[1])    

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

            #-------------         
            #for now test discrete
            TJs = np.array([1.0, 1.8, 2.0, 2.2, 2.25, 2.3, 2.4, 2.6, 3.4])

            shape = (images_count, 1)
            #conditional_labels = np.random.choice(TJs, shape)
            conditional_labels = np.ones(shape, dtype=np.float32)
            conditional_labels[:, 0] = [1.0 , 1.0,  1.8, 1.8,
                                        1.0 , 1.0,  1.8, 1.8,
                                        2.25, 2.25, 3.4, 3.4,
                                        2.25, 2.25, 3.4, 3.4,]

            conditional_labels = tf.repeat(conditional_labels, repeats=[self.model.conditional_dim])
            conditional_labels = tf.reshape(conditional_labels, (images_count, self.model.conditional_dim))

            #-------------
            random_vectors = sample_generator_input(images_count, self.model.latent_dim)
            latent_vectors = tf.concat([random_vectors, conditional_labels], axis=1)

            generated_images = self.model.generator(latent_vectors)
            generated_images = (generated_images + 1.0) / 2.0

            #-------------
            plot_images(generated_images, conditional_labels, images_count, epoch, self.model.plot_path)
