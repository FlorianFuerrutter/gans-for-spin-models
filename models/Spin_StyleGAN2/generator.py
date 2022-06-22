#latent
#FC
#FC
#FC
#out W latent with 4 invidual vectors to inject -> A
#
#const 8x8
#A+WeightDemodulationConv2D
#bias + B
#------------------------------->ToRGB conv1x1 filter 3 + UP
#
#bilin UP 16x16
#A+WeightDemodulationConv2D
#bias + B
#A+WeightDemodulationConv2D
#bias + B
#------------------------------->ToRGB conv1x1 filter 3 + UP + ADD
#
#bilin UP 32x32
#A+WeightDemodulationConv2D
#bias + B
#A+WeightDemodulationConv2D
#bias + B
#------------------------------->ToRGB conv1x1 filter 3 + UP + ADD
#
#bilin UP 64x64
#A+WeightDemodulationConv2D
#bias + B
#A+WeightDemodulationConv2D
#bias + B
#------------------------------->ToRGB conv1x1 filter 3 + UP  + ADD --> OUTPUT!!!!!!!

#--------------------------------------------------------------------
# https://arxiv.org/pdf/1912.04958.pdf
# https://arxiv.org/pdf/1812.04948.pdf
#--------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, initializers
from custom_layers import BiasNoiseBroadcastLayer, Conv2DMod

#--------------------------------------------------------------------

def tRGB_block(inp, style, filter_size, out_filter):
    kernel_initializer = initializers.VarianceScaling(200.0/inp.shape[2])

    rgb_style = layers.Dense(filter_size, kernel_initializer=kernel_initializer)(style)
    rgb_style = layers.Dropout(0.2)(rgb_style)

    out = Conv2DMod(out_filter, 1, kernel_initializer=kernel_initializer, demod=False)([inp, rgb_style])

    return out

def enc_block(enc_input, in_style, noise_image, filter_size, out_filter, kernel_size, kernel_initializer, strides=1, padding='same', first_block=False):
    enc = enc_input
    
    #--------------------------------------------
    #UpSample, not on first_block
    if not first_block:
        #enc = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(enc)
        enc = layers.Conv2DTranspose(filter_size, kernel_size=(4,4), strides=(2,2), padding=padding, kernel_initializer=kernel_initializer)(enc)
        enc = layers.LeakyReLU(0.2)(enc)

    #crop noise to current image size
    cropping = ((noise_image.shape[1]-enc.shape[1], 0), (noise_image.shape[2]-enc.shape[2], 0))
    noise = layers.Cropping2D(cropping=cropping)(noise_image)
    noise = layers.Dense(filter_size, kernel_initializer='zeros')(noise)

    if not first_block:
        style = layers.Dense(enc.shape[-1], kernel_initializer=kernel_initializer)(in_style)
        style = layers.Dropout(0.2)(style)

        enc = Conv2DMod(filter_size, kernel_size, demod=True, strides=strides, kernel_initializer=kernel_initializer, padding=padding)([enc, style])       
        enc = BiasNoiseBroadcastLayer(filter_size)([enc, noise])
        enc = layers.LeakyReLU(0.2)(enc)

    #--------------------------------------------
    #Second block always
    style = layers.Dense(enc.shape[-1], kernel_initializer=kernel_initializer)(in_style)
    style = layers.Dropout(0.2)(style)

    enc = Conv2DMod(filter_size, kernel_size, demod=True, strides=strides, kernel_initializer=kernel_initializer, padding=padding)([enc, style])
    enc = BiasNoiseBroadcastLayer(filter_size)([enc, noise])
    enc = layers.LeakyReLU(0.2)(enc)

    #--------------------------------------------
    #RGB block
    rgb = tRGB_block(enc, in_style, filter_size, out_filter)

    return enc, rgb

#--------------------------------------------------------------------

def create_mapping_network(latent_dim, styles_dim):
    #--------------------------------------------
    #style mapping network
    
    latent_input = layers.Input(shape=latent_dim)

    out = latent_input
    out = layers.Dense(styles_dim, use_bias=True)(out)
    out = layers.LeakyReLU(0.2)(out)

    #out = layers.Dense(styles_dim)(out)
    #out = layers.LeakyReLU(0.2)(out)

    #--------------------------------------------
    map_model = keras.models.Model(inputs=latent_input, outputs=out, name="mapping_network")
    return map_model

def create_generator(enc_block_count, latent_dim, styles_dim, noise_image_res, out_filter):
    init = keras.initializers.GlorotUniform()
    
    filter_size_const = 64
    filter_size_start = 12 #256     #256 #288 #312
    res_start         = 4      #4

    #create mapping operator, z->w
    mapping_model = create_mapping_network(latent_dim, styles_dim)

    #--------------------------------------------
    #inputs latent and noise, (a mapped style for each enc_block)
    latent_input = []
    style_input  = []
    noise_image_input = []

    for i in range(enc_block_count):
        #latent convert
        latent_input.append(layers.Input(shape=latent_dim))
        style_input.append(mapping_model(latent_input[-1]))

        #noise input
        noise_shape = (noise_image_res, noise_image_res, 1)
        noise_image_input.append(layers.Input(shape=noise_shape))

    #--------------------------------------------
    #Model

    #constant for batch size
    c1 = layers.Lambda(lambda x: x[:, :1] * 0 + 1)(style_input[0])
    x = layers.Dense(res_start * res_start * filter_size_const, use_bias=False, activation='relu', kernel_initializer='zeros')(c1)
    x = layers.Reshape((res_start, res_start, filter_size_const))(x) 

    #first block
    x, rgb = enc_block(x, style_input[0], noise_image_input[0], filter_size=filter_size_start, out_filter=out_filter, kernel_size=(3,3), kernel_initializer=init, first_block=True)

    #--------------------------------------------
    #scale blocks
    for i in range(1, enc_block_count):
        #filter_size = filter_size_start - 32 * (i-1)
        #filter_size = filter_size_start / (2**(i-1))
        filter_size = filter_size_start * (2**(i-1))

        x, rgb_c = enc_block(x, style_input[i], noise_image_input[i], filter_size=filter_size, out_filter=out_filter, kernel_size=(3,3), kernel_initializer=init, first_block=False)       
        
        #rgb = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(rgb)
        rgb = layers.Conv2DTranspose(out_filter, kernel_size=(4,4), strides=(2,2), padding="same", kernel_initializer=init)(rgb)
        #rgb = layers.LeakyReLU(0.2)(rgb)
        rgb = layers.Add()([rgb, rgb_c])        
        
    #--------------------------------------------
    #Activation-layer
    #rgb = layers.Conv2D(out_filter, kernel_size=(5,5), strides=1, padding='same')(rgb)
    output = layers.Activation(activations.tanh, dtype="float32")(rgb)

    g_model = keras.models.Model(inputs=[latent_input, noise_image_input], outputs=output, name="generator")
    return g_model

#--------------------------------------------------------------------


####################DIS#####################################
####################DIS#####################################
####################DIS#####################################
####################DIS#####################################


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












