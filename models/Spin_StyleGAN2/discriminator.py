#from re import X
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, initializers
from custom_layers import BiasNoiseBroadcastLayer, Conv2DMod

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

def dec_block(dec_input, filter_size, kernel_size, kernel_initializer, drop_rate=0.0, padding='same', last_block=False):
    dec = dec_input
    
    #--------------------------------------------
    # Residual net block   
    dec = layers.Conv2D(filter_size, kernel_size=1, strides=(1,1), padding=padding, kernel_initializer=kernel_initializer)(dec)   

    #------- res 
    res = layers.Conv2D(filter_size, kernel_size=kernel_size, strides=(1,1), padding=padding, kernel_initializer=kernel_initializer)(dec)
    res = layers.LeakyReLU(alpha=0.2)(res)
    if drop_rate > 0.0:
        res = layers.Dropout(drop_rate)(res)

    res = layers.Conv2D(filter_size, kernel_size=kernel_size, strides=(1,1), padding=padding, kernel_initializer=kernel_initializer)(res)
    res = layers.LeakyReLU(alpha=0.2)(res)
    if drop_rate > 0.0:
        res = layers.Dropout(drop_rate)(res)
   
    #-------
    out = layers.Add()([res, dec])  
    out = layers.Lambda(lambda x: x *  0.707107)(out) #1/sqrt2

    if not last_block:
        out = layers.AveragePooling2D()(out)

    return out

#--------------------------------------------------------------------

def create_discriminator(image_res):
    init = keras.initializers.GlorotUniform() 

    start_filter_size = 16
    drop_rate         = 0.0
    
    #Structure
    image_input = layers.Input(shape=image_res) #64x64

    #fRGB
    x = layers.Conv2D(start_filter_size, kernel_size=1, strides=(1,1), padding='same', kernel_initializer=init)(image_input)

    #-----------Decoders   
    x = dec_block(x,  start_filter_size * 1, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init) #32x32
    x = dec_block(x,  start_filter_size * 2, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init) #16x16
    x = dec_block(x,  start_filter_size * 4, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init) #8x8
    x = dec_block(x,  start_filter_size * 6, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init, last_block=True) #4x4

    #was 16 and *6, drop 0.0

    #----------- Activation-layer
    x = layers.Flatten()(x)  
    #x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation=activations.sigmoid)(x)

    #----------- Model
    d_model = keras.models.Model(inputs=image_input, outputs=output, name="discriminator")
    return d_model

#--------------------------------------------------------------------