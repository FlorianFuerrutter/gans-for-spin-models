from tensorflow import keras
from tensorflow.keras import layers, activations

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

    #-----------Decoder
    x = dec_layer(image_input,  64, kernel_size=(4,4), strides=(2,2), drop_rate=0.3, kernel_initializer=init) 
    x = dec_layer(x          , 128, kernel_size=(4,4), strides=(2,2), drop_rate=0.3, kernel_initializer=init) 
    x = dec_layer(x          , 256, kernel_size=(4,4), strides=(2,2), drop_rate=0.3, kernel_initializer=init) 

    #----------- Activation-layer
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation=activations.sigmoid)(x)

    #----------- Model
    d_model = keras.models.Model(inputs = image_input, outputs = output, name="discriminator")
    return d_model