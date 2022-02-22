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

#    Model / data parameters
#    num_classes = 10
#    input_shape = (28, 28, 1)

#    # the data, split between train and test sets
#    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#    # Scale images to the [0, 1] range
#    x_train = x_train.astype("float32") / 255
#    x_test = x_test.astype("float32") / 255
#    # Make sure images have shape (28, 28, 1)
#    x_train = np.expand_dims(x_train, -1)
#    x_test = np.expand_dims(x_test, -1)
#    print("x_train shape:", x_train.shape)
#    print(x_train.shape[0], "train samples")
#    print(x_test.shape[0], "test samples")

#    # convert class vectors to binary class matrices
#    y_train = keras.utils.to_categorical(y_train, num_classes)
#    y_test = keras.utils.to_categorical(y_test, num_classes)


#    image_input = layers.Input(shape=input_shape)
#    x = layers.Dense(8, activation="softmax")(image_input)
   
#    #--------------------------- 
#    outfeatures = 4#32
#    infeatures  = 3#16

#    batches     = 1
#    res         = 2    

#    g = Conv2DMod(outfeatures, demod=True, kernel_size=1)
#    a = tf.constant(tf.ones((batches, res, res, infeatures)))
#    b = tf.constant(tf.random.uniform((batches, infeatures)))
#    u = g([a, b])
#    #print(u)

#    #-------
    
#    noise = layers.GaussianNoise(0.1)(x)
#    noise = layers.Flatten()(noise)
#    noise = layers.Dense(infeatures, activation="softmax")(noise) 

#    x = layers.Dense(infeatures, activation="softmax")(x)
#    x = Conv2DMod(outfeatures, demod=True, kernel_size=3)([x, noise])
#    x = layers.LeakyReLU(0.2)(x)

#    noise = layers.GaussianNoise(0.1)(x)
#    noise = layers.Flatten()(noise)
#    noise = layers.Dense(outfeatures, activation="softmax")(noise)
#    x = BiasNoiseBroadcastLayer()([x, noise])

#    #---------------------------

#    x = layers.Flatten()(x)
#    x = layers.Dropout(0.2)(x)
#    x = layers.Dense(num_classes, activation="softmax")(x)
#    model = keras.models.Model(inputs = image_input, outputs = x, name="discriminator")
#    #model.summary()

#    batch_size = 128
#    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#    model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_split=0.1)

#-------------------------------------
#-------------------------------------

from custom_layers import BiasNoiseBroadcastLayer, Conv2DMod
from tensorflow import keras
from tensorflow.keras import layers, activations

import gan
import generator
import discriminator

def main() -> int:  
    #--------------
    #setup

    epochs      = 100
    latent_dim  = 256
    style_dim   = 256 #512

    image_size = (64, 64, 3)
    batch_size = 64
    enc_block_count = int(np.log2(image_size[0])-2)


    #--------------
    #load data
    path     = os.path.join(os.path.dirname(__file__), "..", "..", "..", "img_align_celeba_part1")
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs")

    #--------------
    
    if 0:
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

    #--------------

    g_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.99)
    d_optimizer = keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.0, beta_2=0.99)

    k = keras.losses.BinaryCrossentropy()

    #--------------

    g_model = generator.create_generator(enc_block_count, latent_dim, style_dim)
    d_model = discriminator.create_discriminator(image_size)

    #g_model.summary()
    #d_model.summary()s

    gan_model = gan.gan(d_model, g_model, latent_dim)
    gan_model.compile(d_optimizer, g_optimizer, k, k)

    #--------------
    #train
    gan_model.fit(dataset, epochs=epochs, callbacks=[gan.train_callback(latent_dim=latent_dim), keras.callbacks.TensorBoard(log_dir=log_path)])

    return 0

if __name__ == '__main__':
    main()