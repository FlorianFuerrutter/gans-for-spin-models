import tensorflow as tf
from tensorflow import keras
from keras import layers, activations

#--------------------------------------------------------------------

class PeriodicPadding2D(keras.layers.Layer):
    '''Pads a image with periodic conditions'''

    def __init__(self, padding=2, *args, **kwargs):
        super(PeriodicPadding2D, self).__init__(*args, **kwargs)
        self.padding = padding

    #def compute_output_shape(self, input_shape):
    #    shape = tf.shape(input_shape)
    #    assert shape.size == 3  

    #    if shape[1] is not None:
    #      length_h = shape[1] + 2 * self.padding         
    #    else:
    #      length_h = None

    #    if shape[1] is not None:
    #      length_w = shape[2] + 2 * self.padding         
    #    else:
    #      length_w = None

    #    return tuple([shape[0], length_h, length_w])

    def call(self, inputs):  
        x    = inputs
        size = self.padding

        #assumes channel last format!

        #pad cols
        x = tf.concat([x[:, :, -size:], x, x[:, :, 0:size]], axis=2) 

        #pad rows
        x = tf.concat([x[:, -size:, :], x, x[:, 0:size, :]], axis=1)

        return x

    def get_config(self):
        config = super(PeriodicPadding2D, self).get_config()
        config.update({"padding": self.padding})
        return config

#--------------------------------------------------------------------

def dec_layer(dec_input, filter_size, kernel_size, kernel_initializer, strides=2, drop_rate=0.0, padding='same', use_bn=False):   
    dec = dec_input
    dec = layers.Conv2D(filter_size, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer)(dec)

    if use_bn:
        dec = layers.BatchNormalization()(dec)

    dec = layers.LeakyReLU(alpha=0.2)(dec)

    if drop_rate > 0.0:
        dec = layers.Dropout(drop_rate)(dec)

    return dec

def create_discriminator(image_res):
    init = keras.initializers.GlorotUniform() #RandomNormal(stddev = 0.03)

    #Structure
    image_input = layers.Input(shape=image_res)

    x = layers.GaussianNoise(0.01)(image_input)

    #Add periodic bounding conditions
    x = PeriodicPadding2D(padding=1)(x)

    #-----------Decoder
    drop_rate = 0.0
      
    #good all // 1

    x = dec_layer(x,   64//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init, padding='valid') #32x32  #64//2
    x = dec_layer(x,  128//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #16x16                   #128//2
    x = dec_layer(x,  128//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #8x8                     #128//2
    #x = dec_layer(x, 128, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #4x4

    #64//2 #96//2 #128//2, best

    #----------- Activation-layer
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation=activations.sigmoid)(x)

    #----------- Model
    d_model = keras.models.Model(inputs = image_input, outputs = output, name="discriminator")
    return d_model