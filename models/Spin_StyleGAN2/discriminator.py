import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, initializers
from custom_layers import BiasNoiseBroadcastLayer, Conv2DMod

#--------------------------------------------------------------------

def dec_block(dec_input, filter_size, kernel_size, kernel_initializer, drop_rate=0.0, padding='same'):
    dec = dec_input
    
    #--------------------------------------------
    # Residual net block   
    dec = layers.Conv2D(filter_size, kernel_size=1, strides=(1,1), padding=padding, kernel_initializer=kernel_initializer)(dec)   

    #------- res 1
    res1 = layers.AveragePooling2D()(dec)

    #------- res 2
    res2 = layers.Conv2D(filter_size, kernel_size=kernel_size, strides=(2,2), padding=padding, kernel_initializer=kernel_initializer)(dec)
    res2 = layers.LeakyReLU(alpha=0.2)(res2)
    if drop_rate > 0.0:
        res2 = layers.Dropout(drop_rate)(res2)

    res2 = layers.Conv2D(filter_size, kernel_size=kernel_size, strides=(1,1), padding=padding, kernel_initializer=kernel_initializer)(res2)
    res2 = layers.LeakyReLU(alpha=0.2)(res2)
    if drop_rate > 0.0:
        res2 = layers.Dropout(drop_rate)(res2)
   
    #-------
    out = layers.Add()([res1, res2])  
    return out

#--------------------------------------------------------------------

def create_discriminator(image_res):
    init = keras.initializers.GlorotUniform() 

    #Structure
    image_input = layers.Input(shape=image_res) #64x64

    #-----------Decoder
    x = dec_block(image_input,  64, kernel_size=(4,4), drop_rate=0.2, kernel_initializer=init) #32x32
    x = dec_block(x          , 128, kernel_size=(4,4), drop_rate=0.2, kernel_initializer=init) #16x16
    x = dec_block(x          , 256, kernel_size=(4,4), drop_rate=0.2, kernel_initializer=init) #8x8

    #----------- Activation-layer
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation=activations.sigmoid)(x)

    #----------- Model
    d_model = keras.models.Model(inputs=image_input, outputs=output, name="discriminator")
    return d_model

#--------------------------------------------------------------------