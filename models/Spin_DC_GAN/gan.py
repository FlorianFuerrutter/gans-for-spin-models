import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from random import random
import os

import generator
import discriminator
from keras import backend as K

#--------------------------------------------------------------------

def plot_images(generated_images, images_count, epoch, plot_path=""):
    fig = plt.figure(figsize=(5, 5))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.05, hspace=0.05)

    res = int(np.sqrt(images_count))
    for i in range(images_count):
        plt.subplot(res, res, i+1)
        plt.axis('off')

        generated_image = generated_images[i].numpy()
        generated_image = np.where(generated_image < 0.5, -1, 1)
        plt.imshow(generated_image)  
        
    if plot_path == "":
        plt.savefig("img/generated_{epoch}.png".format(epoch=epoch), bbox_inches='tight')
    else:
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        plt.savefig(plot_path + "/generated_{epoch}.png".format(epoch=epoch), bbox_inches='tight')
    plt.close()

def sample_generator_input(batch_size, latent_dim,):
    return tf.random.normal(shape=(batch_size, latent_dim))
  
#--------------------------------------------------------------------
#    real_labels = tf.zeros((batch_size, 1)) 
#    fake_labels = tf.ones((batch_size, 1))

@tf.function
def dis_loss(y_true, y_pred):
    loss = -tf.math.softplus(y_true * y_pred)
    loss = tf.reduce_mean(loss)
    return loss

@tf.function
def gen_loss(y_true, y_pred):
    loss = tf.math.softplus(y_pred)
    loss = tf.reduce_mean(loss)
    return loss

@tf.function
def wasserstein_loss(y_true, y_pred):
    label_smooth = 1 - 0.05
    y_true = (2.0 * label_smooth) * y_true - label_smooth
    y_pred = (2.0 * label_smooth) * y_pred - label_smooth
    return -tf.reduce_mean(y_true * y_pred)

class gan(keras.Model):
    def __init__(self, latent_dim, image_size):
        super().__init__()

        self.generator = generator.create_generator(latent_dim)
        self.discriminator = discriminator.create_discriminator(image_size)

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
        #labels  = tf.concat( [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0 )   
        #labels += 0.05 * tf.random.uniform(tf.shape(labels)) 

        real_labels = tf.zeros((batch_size, 1)) 
        fake_labels = tf.ones((batch_size, 1))

        noisy_real_labels = real_labels + 0.05 * tf.random.uniform(tf.shape(real_labels)) 
        noisy_fake_labels = fake_labels + 0.05 * tf.random.uniform(tf.shape(fake_labels)) 

        #Record operations for automatic differentiation
        #all watched variables within scope
        with tf.GradientTape() as tape: 
            #fake images combined with real ones
            generated_images = self.generator(latent_vectors)
            #combined_images  = tf.concat([generated_images, real_images], axis=0)

            #--------------

            fake_predictions = self.discriminator(generated_images) 
            real_predictions = self.discriminator(real_images) 

            fake_loss = self.d_loss_fn(noisy_fake_labels, fake_predictions) 
            real_loss = self.d_loss_fn(noisy_real_labels, real_predictions)
            d_loss = fake_loss + real_loss

            #loss = tf.math.softplus(fake_predictions) + tf.math.softplus(-real_predictions)
            #loss = tf.reduce_mean(loss)
            #d_loss = loss

            #--------------

            #predictions = self.discriminator(combined_images) 
            #d_loss      = self.d_loss_fn(labels, predictions) 
            
        #derivative of d_loss with respect to trainable_weights
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)

        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
 
        #--------------------------------------------
        #train generator
      
        #latent and noise
        latent_vectors = sample_generator_input(batch_size, self.latent_dim)

        #Record operations for automatic differentiation
        #all watched variables within scope
        with tf.GradientTape() as tape: 
            predictions = self.discriminator(self.generator(latent_vectors)) 
            g_loss      = self.g_loss_fn(real_labels, predictions) 

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