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


#-------------------------------------
#-------------------------------------

from custom_layers import BiasNoiseLayer, Conv2DMod
from tensorflow import keras
from tensorflow.keras import layers, activations

def main() -> int:  
    
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    image_input = layers.Input(shape=input_shape)

    x = layers.Dense(8, activation="softmax")(image_input) 
    
    skip  = BiasNoiseLayer()([x,x])
    noise = layers.GaussianNoise(1)

    print(noise(1))

    x = Conv2DMod(16, latent_inject=noise, kernel_size=3)([x,x])

    

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.models.Model(inputs = image_input, outputs = x, name="discriminator")
    model.summary()

    batch_size = 128
    epochs = 5

    #A = tf.ones(4)
    #b = WeightDemodulationConv2D(4, latent_inject=A, kernel_size=3)
    #b = BiasLayer()
    #x = tf.ones((2, 6, 6, 3))
    #print(b(x))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=5, validation_split=0.1)

    return 0


if __name__ == '__main__':
    main()