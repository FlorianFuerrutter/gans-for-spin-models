import os
import numpy as np
import matplotlib.pyplot as plt

#Tensorflow
import tensorflow as tf
from tensorflow import keras
print("[INFO] Using tf version: " + tf.__version__ )
print("[INFO] Build with gpu support: " + format(tf.test.is_built_with_gpu_support()))

device_name = tf.test.gpu_device_name()  
if device_name != '/device:GPU:0':
   print("[WARNING] GPU device not found, using CPU for inference")
   print(" -> TensorFlow GPU support requires an assortment of drivers and libraries: https://www.tensorflow.org/install/gpu#software_requirements")
else:
    print('[INFO] Found GPU at: {}'.format(device_name))

#-------------------------------------
#-------------------------------------

import gan
import conditional_gan
import generator
import discriminator

#-------------------------------------
#-------------------------------------

def train_model(dataset, epochs, save_period, plot_period, latent_dim, image_size, weights_path="", plot_path=""):
    if 1:
        def sample_plot():
            plot_images = []
            count = 1
            for images in dataset:
                for image in images:
                    image = (image + 1.0) /2.0               
                    plot_images.append(image)
                
                    count +=1
                    if count > 16:
                        gan.plot_images(plot_images, 16, "sample", plot_path)
                        return
        sample_plot()

    #--------------
    #define loss and optimizer
    
    decay_steps = 469.0 * 10.0 #steps/epochs
    lr_schedule_g = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.5e-4,
                                                                decay_steps=decay_steps,
                                                                decay_rate=0.9)
    lr_schedule_d = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.25e-4,
                                                                decay_steps=decay_steps,
                                                                decay_rate=0.9)

    g_optimizer = keras.optimizers.Adam(learning_rate=lr_schedule_g, beta_1=0.0, beta_2=0.9) #1.5e-4
    d_optimizer = keras.optimizers.Adam(learning_rate=lr_schedule_d, beta_1=0.0, beta_2=0.9) #1.25e-4 

    d_loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    g_loss_fn = gan.wasserstein_loss

    #--------------
    #create model
   
    gan_model = gan.gan(latent_dim, image_size)
    gan_model.compile(d_optimizer, g_optimizer, d_loss_fn, g_loss_fn)

    if (weights_path != ""):
        gan_model.save_path = weights_path
    if (plot_path != ""):
        gan_model.plot_path = plot_path

    #--------------
    #train
    callbacks = []
    callbacks.append(gan.train_callback(latent_dim, plot_period=plot_period, save_period=save_period))

    gan_model.fit(dataset, epochs=epochs,callbacks=callbacks)

    return

def train_conditional_model(dataset, epochs, save_period, plot_period, latent_dim, conditional_dim, image_size, weights_path="", plot_path=""):
    if 1:
        def sample_plot():
            for batches in dataset:
                images, labels = batches

                images = (images + 1.0) /2.0               

                conditional_gan.plot_images(images[:16], labels[0:16], 16, "sample", plot_path)
                return
        sample_plot()

    #--------------
    #define loss and optimizer
    
    g_optimizer = keras.optimizers.Adam(learning_rate=1.5e-4 , beta_1=0.0, beta_2=0.9) 
    d_optimizer = keras.optimizers.Adam(learning_rate=1.25e-4, beta_1=0.0, beta_2=0.9)

    d_loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    g_loss_fn = conditional_gan.wasserstein_loss

    #--------------
    #create model
   
    gan_model = conditional_gan.conditional_gan(latent_dim, conditional_dim, image_size)
    gan_model.compile(d_optimizer, g_optimizer, d_loss_fn, g_loss_fn)

    if (weights_path != ""):
        gan_model.save_path = weights_path
    if (plot_path != ""):
        gan_model.plot_path = plot_path

    #--------------
    #train
    callbacks = []
    callbacks.append(conditional_gan.train_callback(latent_dim, plot_period=plot_period, save_period=save_period))

    gan_model.fit(dataset, epochs=epochs,callbacks=callbacks)

    return

#-------------------------------------
#-------------------------------------

def load_spin_data(batch_size, res, path, name="simulation_states_TJ_2.6.txt", amplitude=0.7):
    #create new dataset 
    file_path = os.path.join(path, name)
    
    #states = np.loadtxt(file_path, skiprows=1, dtype=np.float16) #float32
    states = np.load(file_path[:-3]+"npy")

    states = np.reshape(states, ( -1, res, res, 1))
    print("[load_spin_data] Found states:", states.shape[0])

    #scale (+-)1 to (+-)amplitude
    states = (states * amplitude).astype(np.float16)

    dataset = tf.data.Dataset.from_tensor_slices(states)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def load_conditional_spin_data(batch_size, res, path, TJs, amplitude=0.7):
    #create new dataset
    first = 1

    for TJ in TJs:        
        file_path = os.path.join(path, "simulation_states_TJ_{TJ}".format(TJ=TJ))

        states = np.load(file_path + ".npy")
        
        states = np.reshape(states, ( -1, res, res, 1))
        states = states[:10000]

        states = (states * amplitude).astype(np.float32)
        labels = (np.ones((states.shape[0], 1)) * TJ).astype(np.float32)

        if first:
            global_states = states
            global_labels = labels
            first = 0
        else:
            global_states = np.append(global_states, states, axis=0)
            global_labels = np.append(global_labels, labels, axis=0)      

    dataset = tf.data.Dataset.from_tensor_slices((global_states, global_labels))
      
   
    dataset = dataset.shuffle(45000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size) 

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    #test if shuffle inside batch better or worse, order of .batch and .shufle important here

    #test if performance hit of large shuffle buffer, or if all is on ram not vram

    #test setting tf to float16 not f32 !!

    return dataset

#-------------------------------------
#-------------------------------------

def main() -> int:  
    #--------------
    #setup
    epochs     = 1002
    latent_dim = 4096 #256

    image_size = (64, 64, 1)
    batch_size = 64 #128
    
    amplitude  = 0.7 

    #--------------  
    save_period = 3
    plot_period = 3

    conditional = True

    #--------------
    #load data and train
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train")

    if conditional:
        #------------
        conditional_dim = 1
        TJs = np.array([1.0, 1.8, 2.0, 2.2, 2.25, 2.3, 2.4, 2.6, 3.4])

        #------------
        dataset  = load_conditional_spin_data(batch_size, image_size[0], path, TJs, amplitude)

        train_conditional_model(dataset, epochs, save_period, plot_period, latent_dim, conditional_dim, image_size)

    else:
        #------------
        dataset = load_spin_data(batch_size, image_size[0], path, name="simulation_states_TJ_2.25.txt", amplitude=amplitude)
   
        train_model(dataset, epochs, save_period, plot_period, latent_dim, image_size)

    #--------------
    return 0

if __name__ == '__main__':
    main()