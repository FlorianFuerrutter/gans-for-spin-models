#from re import X
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, initializers
import custom_layers

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
    dec1 = layers.Conv2D(filter_size, kernel_size=1, strides=(1,1), padding=padding, kernel_initializer=kernel_initializer)(dec)   

    #------- res 
    res = dec_layer(dec, filter_size, kernel_size, kernel_initializer, strides=(1,1), drop_rate=drop_rate, padding=padding)
    res = dec_layer(res, filter_size, kernel_size, kernel_initializer, strides=(1,1), drop_rate=drop_rate, padding=padding)
 
    #-------
    out = layers.Add()([res, dec1])  
    out = layers.Lambda( lambda x: x * tf.math.rsqrt(2.0), dtype="float32")(out) #1/sqrt2
    #out = layers.LeakyReLU(alpha=0.2)(out)

    if not last_block:
        out = layers.AveragePooling2D()(out)

    return out

#--------------------------------------------------------------------

def decoder(x, init):
    start_filter_size = 16 # 24 #24  #32 #18
    drop_rate         = 0.0

    #fRGB
    x = layers.Conv2D(start_filter_size, kernel_size=1, strides=(1,1), padding='valid', kernel_initializer=init)(x)

    #-----------Decoders   
    #x = dec_block(x,  start_filter_size * 1, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init) #32x32
    #x = dec_block(x,  start_filter_size * 2, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init) #16x16
    #x = dec_block(x,  start_filter_size * 4, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init) #8x8
    #x = dec_block(x,  start_filter_size * 8, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init, last_block=True) 

    x = dec_block(x,  start_filter_size * 1, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init) #32x32
    x = dec_block(x,  start_filter_size * 2, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init) #16x16
    #x = dec_block(x,  start_filter_size * 4, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init) #8x8
    x = dec_block(x,  start_filter_size * 3, kernel_size=(3,3), drop_rate=drop_rate, kernel_initializer=init, last_block=True) 

    return x

#--------------------------------------------------------------------

def create_discriminator(image_res, cond_channels=0, use_aux=False):
    init = keras.initializers.GlorotUniform() 
   
    #Structure
    input_shape = (image_res[0], image_res[1], image_res[2] + cond_channels)
    image_input = layers.Input(shape=input_shape) #64x64

    if use_aux:
        A_model = create_A_model(image_res, init)
        A_in = image_input[:, :, :, 0:image_res[2]]
        output_A = A_model(A_in)

    #-----------Decoder
    x = custom_layers.PeriodicPadding2D(padding=1)(image_input)
    x = decoder(x, init)

    #----------- Activation-layer
    x = layers.Flatten()(x)  
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)
    output_dis = layers.Activation(activations.sigmoid, dtype="float32")(x)

    #----------- Model
    outputs = [output_dis, output_A] if use_aux else output_dis

    d_model = keras.models.Model(inputs=image_input, outputs=outputs, name="discriminator")
    if use_aux:
        return d_model, A_model

    return d_model, None

#--------------------------------------------------------------------

#--------------------------------------------------------------------
#--------------------------------------------------------------------

def decoderA(x, init):
    #-----------Drop rate
    drop_rate = 0.0
    
    def dec_layerA(dec_input, filter_size, kernel_size, kernel_initializer, strides=2, drop_rate=0.0, padding='same', use_bn=False):   
        dec = dec_input
        dec = layers.Conv2D(filter_size, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer)(dec)

        if use_bn:
            dec = layers.BatchNormalization()(dec)

        dec = layers.LeakyReLU(alpha=0.2)(dec)

        if drop_rate > 0.0:
            dec = layers.Dropout(drop_rate)(dec)

        return dec

    #-----------Decoder
    x = dec_layerA(x,   64//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init, padding='valid') #32x32  #64//2
    x = dec_layerA(x,  128//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #16x16                   #128//2
    x = dec_layerA(x,  196//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #8x8                     #128//2
    #x = dec_layer(x, 128, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #4x4

    #-----------
    return x

def create_A_model(image_res, init):

    image_input = layers.Input(shape=image_res)

     #-----------Decoder
    x = custom_layers.PeriodicPadding2D(padding=1)(image_input)
    x = decoderA(x, init)

    #----------- Activation-layer
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)
    output = layers.Activation(activations.relu, dtype="float32")(x)

    #----------- Model
    model = keras.models.Model(inputs = image_input, outputs = output, name="A_model")
    return model