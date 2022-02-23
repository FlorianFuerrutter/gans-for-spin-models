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

def tRGB_block(inp, style, filter_size):
    kernel_initializer = initializers.VarianceScaling(200/inp.shape[2])

    rgb_style = layers.Dense(filter_size, kernel_initializer=kernel_initializer)(style)
    out = Conv2DMod(3, 1, kernel_initializer=kernel_initializer, demod=False)([inp, rgb_style])

    return out

def enc_block(enc_input, in_style, filter_size, kernel_size, kernel_initializer, strides=1, padding='same', first_block=False):
    enc = enc_input
    
    #--------------------------------------------
    #UpSample not on first_block
    if not first_block:
        style = layers.Dense(enc.shape[-1], kernel_initializer=kernel_initializer)(in_style)

        enc = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(enc)
        enc = Conv2DMod(filter_size, kernel_size, demod=True, strides=strides, kernel_initializer=kernel_initializer, padding=padding)([enc, style])
        enc = BiasNoiseBroadcastLayer(filter_size)(enc)
        enc = layers.GaussianNoise(0.01)(enc)
        enc = layers.LeakyReLU(0.2)(enc)

    #--------------------------------------------
    #Second block always
    style = layers.Dense(enc.shape[-1], kernel_initializer=kernel_initializer)(in_style)
    
    enc = Conv2DMod(filter_size, kernel_size, demod=True, strides=strides, kernel_initializer=kernel_initializer, padding=padding)([enc, style])
    enc = BiasNoiseBroadcastLayer(filter_size)(enc)
    enc = layers.GaussianNoise(0.01)(enc)
    enc = layers.LeakyReLU(0.2)(enc)

    #--------------------------------------------
    #RGB block
    rgb = tRGB_block(enc, in_style, filter_size)

    return enc, rgb

#--------------------------------------------------------------------

def create_mapping_network(latent_dim, style_dim):
    #--------------------------------------------
    #style mapping network
    
    latent_input = layers.Input(shape=latent_dim)

    out = latent_input
    out = layers.Dense(style_dim)(out)
    out = layers.LeakyReLU(0.2)(out)

    out = layers.Dense(style_dim)(out)
    out = layers.LeakyReLU(0.2)(out)

    out = layers.Dense(style_dim)(out)
    out = layers.LeakyReLU(0.2)(out)
    
    #out = layers.Dense(style_dim)(out)
    #out = layers.LeakyReLU(0.2)(out)

    #--------------------------------------------
    map_model = keras.models.Model(inputs=latent_input, outputs=out, name="mapping_network")
    return map_model

def create_generator(enc_block_count, latent_dim, style_dim):
    init = keras.initializers.GlorotUniform()

    map_net = create_mapping_network(latent_dim, style_dim)

    #--------------------------------------------
    #inputs, a mapped style for each enc_block
    latent_input = layers.Input(shape=(latent_dim))

    style_input = map_net(latent_input)

    #--------------------------------------------
    #Model
    filter_size_start = 256
    res_start         = 8

    x = layers.Dense(res_start * res_start * filter_size_start, use_bias=False, activation='relu')(style_input)
    x = layers.Reshape((res_start, res_start, filter_size_start))(x) 
    x, rgb = enc_block(x, style_input, filter_size=filter_size_start, kernel_size=(3,3), kernel_initializer=init, first_block=True)

    for i in range(1, enc_block_count):    
        filter_size = filter_size_start / (2**i)

        x, rgb_c = enc_block(x, style_input, filter_size=filter_size, kernel_size=(3,3), kernel_initializer=init, first_block=False)       
        
        rgb = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(rgb)
        rgb = layers.Add()([rgb, rgb_c])        
        
    #--------------------------------------------
    #Activation-layer
    output = layers.Conv2D(3, kernel_size=(5,5), strides=1, padding='same', activation=activations.tanh)(rgb)

    g_model = keras.models.Model(inputs=latent_input, outputs=output, name="generator")
    return g_model

#--------------------------------------------------------------------