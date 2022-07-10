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

def dec_block(x, init, image_size):
    #-----------Drop rate
    drop_rate = 0.0
    scale = 2  

    #-----------Decoder
    x = dec_layer(x,   64//scale, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init, padding='valid') #32x32  #64//2
    x = dec_layer(x,  128//scale, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #16x16                   #128//2
    x = dec_layer(x,  196//scale, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #8x8                     #128//2
    #x = dec_layer(x, 128, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #4x4

    #-----------
    return x

#--------------------------------------------------------------------

def create_discriminator(image_size, cond_channels=0, use_aux=False):
    init = keras.initializers.GlorotUniform() #RandomNormal(stddev = 0.03)

    #Structure
    dis_input_shape = (image_size[0], image_size[1], image_size[2] + cond_channels)
    image_input = layers.Input(shape=dis_input_shape)
   
    if use_aux:
        A_model = create_A_model(image_size)
        A_in = image_input[:, :, :, 0:image_size[2]]
        output_A = A_model(A_in)

    #Add periodic bounding conditions
    #x = layers.GaussianNoise(0.01)(image_input)
    x = PeriodicPadding2D(padding=1)(image_input)
    
    #-----------Decoder
    x = dec_block(x, init, image_size)

    #----------- Activation-layer
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    output_dis = layers.Dense(1, activation=activations.sigmoid)(x)

    #----------- Model

    outputs = [output_dis, output_A] if use_aux else output_dis

    d_model = keras.models.Model(inputs = image_input, outputs = outputs, name="discriminator")

    if not use_aux:
        return d_model
    return d_model, A_model

def create_A_model(image_res):
    init = keras.initializers.GlorotUniform() #RandomNormal(stddev = 0.03)

    #Structure
    image_input = layers.Input(shape=image_res)

    #Add periodic bounding conditions
    x = PeriodicPadding2D(padding=1)(image_input)

    #-----------Decoder
    x = dec_block(x, init, image_res)

    #----------- Activation-layer
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation=activations.relu)(x)

    #----------- Model
    d_model = keras.models.Model(inputs = image_input, outputs = output, name="A_model")
    return d_model