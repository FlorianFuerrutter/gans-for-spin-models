import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class gan(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator     = generator
        self.latent_dim    = latent_dim

        #fixed loss metrix here
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

        #fake images combined with real ones
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))   
        
        generated_images = self.generator(random_latent_vectors)
        combined_images  = tf.concat([generated_images, real_images], axis=0)
       
        #set labels
        labels  = tf.concat( [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0 )   
        labels += 0.05 * tf.random.uniform(tf.shape(labels)) 

        #Record operations for automatic differentiation
        #all watched variables within scope
        with tf.GradientTape() as tape: 
            predictions = self.discriminator(combined_images) 
            d_loss      = self.d_loss_fn(labels, predictions) 

            #derivative of d_loss with respect to trainable_weights
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)

            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
 
        #------------------------------
        #train generator

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim)) 
        misleading_labels = tf.zeros((batch_size, 1)) 
 
        with tf.GradientTape() as tape: 
            predictions = self.discriminator(self.generator(random_latent_vectors)) 
            g_loss      = self.g_loss_fn(misleading_labels, predictions) 

            #derivative of g_loss with respect to trainable_weights
            grads = tape.gradient(g_loss, self.generator.trainable_weights) 

            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights)) 
 
        #------------------------------

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

class train_callback(keras.callbacks.Callback):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        if ( (epoch % 1) != 0 ):
            return

        images = 9

        random_latent_vectors = tf.random.normal(shape=(images, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images + 1.0) / 2.0

        fig = plt.figure(figsize=(6, 6))
        for i in range(images):
            plt.subplot(3, 3, i+1)
            plt.axis('off')
            plt.imshow(generated_images[i].numpy())           
        plt.savefig("img/generated_{epoch}.png".format(epoch=epoch), bbox_inches='tight')
