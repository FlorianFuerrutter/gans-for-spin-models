import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from random import random

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

    #noise styles
    #for i in range(enc_block_count):
    #    latent_vectors.append(latent_vector())

    #direct noise inputs
    noise_images   = []
    for i in range(enc_block_count):
        noise_images.append(tf.random.normal(shape=(batch_size, noise_image_res, noise_image_res, 1)))

    return latent_vectors, noise_images

class gan(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, style_dim, enc_block_count, noise_image_res):
        super().__init__()
        self.discriminator = discriminator
        self.generator     = generator

        #----------
        self.latent_dim      = latent_dim
        self.style_dim       = style_dim
        self.enc_block_count = enc_block_count
        self.noise_image_res = noise_image_res

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
       
        #------------------------------
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
 
        #------------------------------
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
 
        #------------------------------

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

class train_callback(keras.callbacks.Callback):
    def __init__(self, enc_block_count, latent_dim, noise_image_res):
        self.enc_block_count = enc_block_count
        self.latent_dim      = latent_dim
        self.noise_image_res = noise_image_res

    def on_epoch_end(self, epoch, logs=None):
        if ( (epoch % 1) != 0 ):
            return

        images = 9

        latent_vectors, noise_images = sample_generator_input(images, self.enc_block_count, self.latent_dim, self.noise_image_res, t=0.0)
        generated_images = self.model.generator([latent_vectors, noise_images])
        generated_images = (generated_images + 1.0) / 2.0

        fig = plt.figure(figsize=(6, 6))
        for i in range(images):
            plt.subplot(3, 3, i+1)
            plt.axis('off')
            plt.imshow(generated_images[i].numpy())           
        plt.savefig("img/generated_{epoch}.png".format(epoch=epoch), bbox_inches='tight')