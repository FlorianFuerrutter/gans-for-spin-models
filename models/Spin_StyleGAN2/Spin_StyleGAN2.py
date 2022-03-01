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

#    Model / data parameters
#    input_shape = (28, 28, 1)
#
#    # the data, split between train and test sets
#    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#
#    # Scale images to the [0, 1] range
#    x_train = x_train.astype("float32") / 255
#    x_test = x_test.astype("float32") / 255
#    # Make sure images have shape (28, 28, 1)
#    x_train = np.expand_dims(x_train, -1)
#    x_test = np.expand_dims(x_test, -1)
#    #--------------------------- 
#    outfeatures = 4#32
#    infeatures  = 3#16
#
#    batches     = 1
#    res         = 2    
#
#    g = Conv2DMod(outfeatures, demod=True, kernel_size=1)
#    a = tf.constant(tf.ones((batches, res, res, infeatures)))
#    b = tf.constant(tf.random.uniform((batches, infeatures)))
#    u = g([a, b])
#    #print(u)

#-------------------------------------
#-------------------------------------

def load_spin_data(batch_size, res, path, name="simulation_states_TJ_2.5.txt", amplitude=0.9):
    #create and store new dataset 
    file_path = os.path.join(path, name)
    states = np.loadtxt(file_path, skiprows=1, dtype=np.float32)
    states = np.reshape(states, ( -1, res, res, 1))
    print("found states:", states.shape[0])

    #scale (+-)1 to (+-)amplitude
    states = states * amplitude

    dataset = tf.data.Dataset.from_tensor_slices(states)
    dataset = dataset.batch(batch_size)

    return dataset

def main() -> int:  
    #--------------------------------------------------------------------
    #setup
  
    latent_dim  = 256
    style_dim   = 512 
    batch_size  = 128

    epochs      = 1000
    image_size = (64, 64, 3)
    
    enc_block_count = int(np.log2(image_size[0])-1)
    noise_image_res = image_size[0]

    #--------------------------------------------------------------------
    #load data
    path     = os.path.join(os.path.dirname(__file__), "..", "..", "..", "img_align_celeba_part1")
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs")

    #--------------
    
    if 1:
        #if existing dataset, use that
        dataset = tf.data.experimental.load(path)
    else:
        #create and store new dataset
        dataset = tf.keras.utils.image_dataset_from_directory(
                        path,
                        label_mode=None, 
                        image_size=(image_size[0], image_size[1]),
                        batch_size=batch_size,
                        smart_resize=True)
        dataset = dataset.map(lambda x: (x - 127.5) / 127.5)    
        tf.data.experimental.save(dataset, path) 

    #--------------------------------------------------------------------
    #create model

    g_optimizer = keras.optimizers.Adam(learning_rate=2e-3, beta_1=0.0, beta_2=0.9, epsilon=1e-08)
    d_optimizer = keras.optimizers.Adam(learning_rate=2e-3, beta_1=0.0, beta_2=0.9, epsilon=1e-08)
    loss = keras.losses.BinaryCrossentropy(label_smoothing=0.05)
        
    plot_period = 1 
    save_period = 24

    gan_model = gan.gan(enc_block_count, latent_dim, style_dim, image_size, noise_image_res)
    gan_model.compile(d_optimizer, g_optimizer, loss, loss)

    #gan_model.plot_print_model_config()

    #--------------------------------------------------------------------
    #train   

    callbacks = []
    callbacks.append(gan.train_callback(enc_block_count, latent_dim, noise_image_res, plot_period=plot_period, save_period=save_period))
    #callbacks.append(keras.callbacks.TensorBoard(log_dir=log_path))

    gan_model.fit(dataset, epochs=epochs, callbacks=callbacks)

    #store final weights
    gan_model.save(epochs+1, only_weights=True)

    return 0

if __name__ == '__main__':
    main()