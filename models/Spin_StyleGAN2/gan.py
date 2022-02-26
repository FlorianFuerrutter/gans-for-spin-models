import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from random import random

from custom_layers import BiasNoiseBroadcastLayer, Conv2DMod
import generator
import discriminator

#--------------------------------------------------------------------

def plot_images(generated_images, images_count, epoch):
    fig = plt.figure(figsize=(5, 5))
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)

    res = int(np.sqrt(images_count))
    for i in range(images_count):
        plt.subplot(res, res, i+1)
        plt.axis('off')
        plt.imshow(generated_images[i].numpy())                 
    plt.savefig("img/generated_{epoch}.png".format(epoch=epoch), bbox_inches='tight')
    plt.close()

def sample_generator_input(batch_size, enc_block_count, latent_dim, noise_image_res, t= 0.9):
    #latent_vectors will be mapped into styles
    def latent_vector():
        return tf.random.normal(shape=(batch_size, latent_dim))

    if random() < t:
        #use mixed style inputs (2 styles total)  
        part = int(random() * enc_block_count)

        latent1 = [latent_vector()] * part
        latent2 = [latent_vector()] * (enc_block_count-part)

        latent_vectors = latent1 + latent2
    else:
        #use single style inputs (1 style total)  
        latent_vectors = [latent_vector()] * enc_block_count

    #direct noise inputs
    noise_images   = []
    for i in range(enc_block_count):
        #noise_images.append(tf.random.uniform(shape=(batch_size, noise_image_res, noise_image_res, 1)))
        noise_images.append(tf.random.normal(shape=(batch_size, noise_image_res, noise_image_res, 1)))

    return latent_vectors, noise_images

#--------------------------------------------------------------------

class gan(keras.Model):
    def __init__(self, enc_block_count, latent_dim, style_dim, image_size, noise_image_res):
        super().__init__()

        self.generator = generator.create_generator(enc_block_count, latent_dim, style_dim, noise_image_res)
        self.discriminator = discriminator.create_discriminator(image_size)

        #--------------------------------------------
        self.enc_block_count = enc_block_count
        self.latent_dim      = latent_dim
        self.style_dim       = style_dim
        self.image_size      = image_size
        self.noise_image_res = noise_image_res

        self.save_path = "./model-saves/gan_" 

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
        latent_vectors, noise_images = sample_generator_input(batch_size, self.enc_block_count, self.latent_dim, self.noise_image_res)

        #set labels
        labels  = tf.concat( [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0 )   
        labels += 0.05 * tf.random.uniform(tf.shape(labels)) 

        #Record operations for automatic differentiation
        #all watched variables within scope
        with tf.GradientTape() as tape: 
            #fake images combined with real ones
            generated_images = self.generator([latent_vectors, noise_images])
            combined_images  = tf.concat([generated_images, real_images], axis=0)

            #--------------

            predictions = self.discriminator(combined_images) 
            d_loss      = self.d_loss_fn(labels, predictions) 

            #derivative of d_loss with respect to trainable_weights
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)

            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
 
        #--------------------------------------------
        #train generator
      
        #latent and noise
        latent_vectors, noise_images = sample_generator_input(batch_size, self.enc_block_count, self.latent_dim, self.noise_image_res)

        misleading_labels = tf.zeros((batch_size, 1)) 

        #Record operations for automatic differentiation
        #all watched variables within scope
        with tf.GradientTape() as tape: 
            predictions = self.discriminator(self.generator([latent_vectors, noise_images])) 
            g_loss      = self.g_loss_fn(misleading_labels, predictions) 

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
        #custom_objects = {"Conv2DMod": Conv2DMod, "BiasNoiseBroadcastLayer": BiasNoiseBroadcastLayer}
        #model = keras.models.model_from_json(json_config, custom_objects)
        #model = keras.Model.from_config(config, custom_objects)

        if only_weights:
            self.generator.load_weights(self.save_path + "generator_weights_{epoch}".format(epoch=epoch))
            self.discriminator.load_weights(self.save_path + "discriminator_weights_{epoch}".format(epoch=epoch))

        else: 
            self.generator = keras.models.load_model(self.save_path + "generator_full_{epoch}".format(epoch=epoch))
            self.discriminator = keras.models.load_model(self.save_path + "discriminator_full_{epoch}".format(epoch=epoch))

    def plot_print_model_config():
        self.generator.summary()
        self.discriminator.summary()

        tf.keras.utils.plot_model(self.generator, "generator.png", show_shapes=True)
        tf.keras.utils.plot_model(self.discriminator, "discriminator.png", show_shapes=True)

class train_callback(keras.callbacks.Callback):
    def __init__(self, enc_block_count, latent_dim, noise_image_res, plot_period=1, save_period=5):
        super(train_callback, self).__init__()
        self.enc_block_count = enc_block_count
        self.latent_dim      = latent_dim
        self.noise_image_res = noise_image_res

        self.plot_period = plot_period
        self.save_period = save_period

    def on_epoch_end(self, epoch, logs=None):
        
        #--------------------------------------------
        #save weights
        if ((epoch % self.save_period) == 0) and (epoch > 2 * self.save_period):
            self.model.save(epoch, only_weights=True)

        #--------------------------------------------
        #plot images
        if (epoch % self.plot_period) == 0:
            images_count = 16

            latent_vectors, noise_images = sample_generator_input(images_count, self.enc_block_count, self.latent_dim, self.noise_image_res, t=0.0)
            generated_images = self.model.generator([latent_vectors, noise_images])
            generated_images = (generated_images + 1.0) / 2.0

            plot_images(generated_images, images_count, epoch)