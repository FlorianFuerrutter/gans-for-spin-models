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

    g_optimizer = keras.optimizers.Adam(learning_rate=5e-4, beta_1=0.0, beta_2=0.9) 
    d_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.9)
   
    #loss = keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    #loss = gan.wasserstein_loss

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

#-------------------------------------
#-------------------------------------

def load_spin_data(batch_size, res, path, name="simulation_states_TJ_2.6.txt", amplitude=0.9):
    #create and store new dataset 
    file_path = os.path.join(path, name)
    states = np.loadtxt(file_path, skiprows=1, dtype=np.float16) #float32
    states = np.reshape(states, ( -1, res, res, 1))
    print("[load_spin_data] Found states:", states.shape[0])

    #scale (+-)1 to (+-)amplitude
    states = states * amplitude

    dataset = tf.data.Dataset.from_tensor_slices(states)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def main() -> int:  
    #--------------
    #setup
    epochs     = 1002
    latent_dim = 256 #256

    image_size = (64, 64, 1)
    batch_size = 128 #128
    
    amplitude  = 0.7 

    #--------------
    #load data
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train")

    if 1: 
        #create and store new dataset 
        dataset = load_spin_data(batch_size, image_size[0], path, name="simulation_states_TJ_2.25.txt", amplitude=amplitude)    
        #tf.data.experimental.save(dataset, path) 
        #exit(0)
    else:
        #if existing dataset, use that
        dataset = tf.data.experimental.load(path)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    #--------------  
    save_period = 3
    plot_period = 3

    train_model(dataset, epochs, save_period, plot_period, latent_dim, image_size)

    #--------------
    return 0

if __name__ == '__main__':
    main()