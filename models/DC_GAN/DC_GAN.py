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

def main() -> int:  
    #--------------
    #setup
    latent_dim = 128
    image_size = (64, 64, 3)
    batch_size = 410

    epochs     = 20 

    #--------------
    #load data

    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "img_align_celeba_part")
    

    dataset = tf.keras.utils.image_dataset_from_directory(
         path,
         label_mode=None, 
         image_size=(image_size[0], image_size[1]),
         batch_size=batch_size,
         smart_resize=True)
    dataset = dataset.map(lambda x: (x - 127.5) / 127.5)

    #--------------
    #define loss and optimizer

    def g_loss_fn(y_true, y_pred):
        # a logit,         (i.e, value in [-inf, inf] when from_logits=True ) 
        # or a probability (i.e, value in [0.,    1.] when from_logits=False)
        keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True, label_smoothing=0.0)
    def d_loss_fn(y_true, y_pred):
        # a logit,         (i.e, value in [-inf, inf] when from_logits=True ) 
        # or a probability (i.e, value in [0.,    1.] when from_logits=False)
        keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0)

    g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
   
    #--------------
    #create model

    g_model   = generator.create_generator(latent_dim)
    d_model   = discriminator.create_discriminator(image_size)

    k = keras.losses.BinaryCrossentropy()

    gan_model = gan.gan(d_model, g_model, latent_dim)
    gan_model.compile(d_optimizer, g_optimizer, k, k)

    #--------------
    #train
    gan_model.fit(dataset, epochs=epochs,callbacks=[gan.train_callback(latent_dim=latent_dim)])

    return 0


if __name__ == '__main__':
    main()