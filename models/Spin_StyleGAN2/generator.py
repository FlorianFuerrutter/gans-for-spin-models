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
    out = Conv2DMod(out_filter, 1, kernel_initializer=kernel_initializer, demod=False)([inp, rgb_style])

    return out

def enc_block(enc_input, in_style, noise_image, filter_size, out_filter, kernel_size, kernel_initializer, strides=1, padding='same', first_block=False):
    enc = enc_input
    
    #--------------------------------------------
    #UpSample, not on first_block
    if not first_block:
        #enc = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(enc)
        enc = layers.Conv2DTranspose(filter_size, kernel_size=(4,4), strides=(2,2), padding=padding, kernel_initializer=kernel_initializer)(enc)

    cropping = ((noise_image.shape[1]-enc.shape[1], 0), (noise_image.shape[2]-enc.shape[2], 0))
    noise = layers.Cropping2D(cropping=cropping)(noise_image)
    noise = layers.Dense(filter_size, kernel_initializer='zeros')(noise)

    if not first_block:
        style = layers.Dense(enc.shape[-1], kernel_initializer=kernel_initializer)(in_style)
        
        enc = Conv2DMod(filter_size, kernel_size, demod=True, strides=strides, kernel_initializer=kernel_initializer, padding=padding)([enc, style])       
        enc = BiasNoiseBroadcastLayer(filter_size)([enc, noise])
        enc = layers.LeakyReLU(0.2)(enc)

    #--------------------------------------------
    #Second block always
    style = layers.Dense(enc.shape[-1], kernel_initializer=kernel_initializer)(in_style)
    
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
    out = layers.Dense(styles_dim)(out)
    out = layers.LeakyReLU(0.2)(out)

    out = layers.Dense(styles_dim)(out)
    out = layers.LeakyReLU(0.2)(out)

    #out = layers.Dense(styles_dim)(out)
    #out = layers.LeakyReLU(0.2)(out)
    
    #out = layers.Dense(styles_dim)(out)
    #out = layers.LeakyReLU(0.2)(out)

    #--------------------------------------------
    map_model = keras.models.Model(inputs=latent_input, outputs=out, name="mapping_network")
    return map_model

def create_generator(enc_block_count, latent_dim, styles_dim, noise_image_res, out_filter):
    init = keras.initializers.GlorotUniform()
    
    filter_size_const = 64
    filter_size_start = 256     #256 #288 #312
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
        #filter_size = filter_size_start - 64 * (i-1)
        filter_size = filter_size_start / (2**i)

        x, rgb_c = enc_block(x, style_input[i], noise_image_input[i], filter_size=filter_size, out_filter=out_filter, kernel_size=(3,3), kernel_initializer=init, first_block=False)       
        
        #rgb = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(rgb)
        rgb = layers.Conv2DTranspose(out_filter, kernel_size=(4,4), strides=(2,2), padding="same", kernel_initializer=init)(rgb)
        rgb = layers.Add()([rgb, rgb_c])        
        
    #--------------------------------------------
    #Activation-layer
    rgb = layers.Conv2D(out_filter, kernel_size=(5,5), strides=1, padding='same')(rgb)
    output = layers.Activation(activations.tanh, dtype="float32")(rgb)

    g_model = keras.models.Model(inputs=[latent_input, noise_image_input], outputs=output, name="generator")
    return g_model

#--------------------------------------------------------------------