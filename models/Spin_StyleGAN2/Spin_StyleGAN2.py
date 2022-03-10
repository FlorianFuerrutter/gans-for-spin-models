import os
import numpy as np
import matplotlib.pyplot as plt
import gan

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

    #--------------------------------------------------------------------
    #setup
    style_dim  = 512 

    enc_block_count = int(np.log2(image_size[0])-1)
    noise_image_res = image_size[0]

    #--------------------------------------------------------------------
    #create model

    g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.0, beta_2=0.9, epsilon=1e-08)
    d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.0, beta_2=0.9, epsilon=1e-08)
    
    #loss = keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    d_loss_fn = keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    g_loss_fn = gan.wasserstein_loss


    gan_model = gan.gan(enc_block_count, latent_dim, style_dim, image_size, noise_image_res)
    gan_model.compile(d_optimizer, g_optimizer, d_loss_fn, g_loss_fn)

    if (weights_path != ""):
        gan_model.save_path = weights_path
    if (plot_path != ""):
        gan_model.plot_path = plot_path

    #gan_model.plot_print_model_config()

    #--------------------------------------------------------------------
    #train   

    callbacks = []
    callbacks.append(gan.train_callback(enc_block_count, latent_dim, noise_image_res, plot_period=plot_period, save_period=save_period))
    
    #log_path = os.path.join(os.path.dirname(__file__), "..", "logs")
    #callbacks.append(keras.callbacks.TensorBoard(log_dir=log_path))

    gan_model.fit(dataset, epochs=epochs, callbacks=callbacks)

    #store final weights
    gan_model.save(epochs+1, only_weights=True)

    return

#-------------------------------------
#-------------------------------------

def load_spin_data(batch_size, res, path, name="simulation_states_TJ_2.6.txt", amplitude=0.9):
    #create and store new dataset 
    file_path = os.path.join(path, name)
    states = np.loadtxt(file_path, skiprows=1, dtype=np.float32)
    states = np.reshape(states, ( -1, res, res, 1))
    print("found states:", states.shape[0])

    #scale (+-)1 to (+-)amplitude
    states = states * amplitude

    dataset = tf.data.Dataset.from_tensor_slices(states)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def main() -> int:  
    #--------------------------------------------------------------------
    #setup 
    epochs     = 101
    latent_dim = 256  

    image_size = (64, 64, 3) #1 for spin #3 for rgb
    batch_size = 128  #64 #128 #256 

    amplitude   = 0.7

    #--------------------------------------------------------------------
    #load data
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "img_align_celeba_part1")
    
    #--------------
    
    if 1:
        #if existing dataset, use that
        dataset = tf.data.experimental.load(path)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    elif 0:
        #spin
        path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train")
        dataset = load_spin_data(batch_size, image_size[0], path, name="simulation_states_TJ_2.25.txt", amplitude=amplitude) 
    elif 0:
        #create and store new dataset
        dataset = tf.keras.utils.image_dataset_from_directory(
                        path,
                        label_mode=None, 
                        image_size=(image_size[0], image_size[1]),
                        batch_size=batch_size,
                        smart_resize=True)
        dataset = dataset.map(lambda x: (x - 127.5) / 127.5)    
        #tf.data.experimental.save(dataset, path) 
    elif 0: #mnist
        #load
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
                  
        #28x28 to 64x64 with padding, (64-28)/2=18
        x_train = np.stack((x_train,) * 3, axis=-1).astype("float32")
        images = tf.image.pad_to_bounding_box(x_train, 18, 18, 64, 64)  

        #convert to Dataset
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.map(lambda x: (x - 127.5) / 127.5)    
        dataset = dataset.batch(batch_size)

    #--------------
    plot_period = 1
    save_period = 300

    train_model(dataset, epochs, save_period, plot_period, latent_dim, image_size)

    #--------------
    return 0

if __name__ == '__main__':
    main()