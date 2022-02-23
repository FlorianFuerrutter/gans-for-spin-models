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

def dec_block(dec_input, filter_size, kernel_size, kernel_initializer, drop_rate=0.0, padding='same'):
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
    out = layers.AveragePooling2D()(out)

    return out

#--------------------------------------------------------------------

def create_discriminator(image_res):
    init = keras.initializers.GlorotUniform() 

    #Structure
    image_input = layers.Input(shape=image_res) #64x64

    #-----------Decoders
    x = dec_layer(image_input,  64, kernel_size=(3,3), drop_rate=0.0, kernel_initializer=init) #32x32
    x = dec_layer(x          , 128, kernel_size=(3,3), drop_rate=0.0, kernel_initializer=init) #16x16
    x = dec_layer(x          , 256, kernel_size=(3,3), drop_rate=0.0, kernel_initializer=init) #8x8
    #x = dec_block(x          , 512, kernel_size=(3,3), drop_rate=0.0, kernel_initializer=init) #4x4

    #----------- Activation-layer
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation=activations.sigmoid)(x)

    #----------- Model
    d_model = keras.models.Model(inputs=image_input, outputs=output, name="discriminator")
    return d_model

#--------------------------------------------------------------------