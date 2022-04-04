from tensorflow import keras
from tensorflow.keras import layers, activations

def enc_layer(enc_input, filter_size, kernel_size, kernel_initializer, strides=2, drop_rate=0.0, padding='same', use_bn=False):   
    enc = enc_input
    enc = layers.Conv2DTranspose(filter_size, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer)(enc)

    if use_bn:
        enc = layers.BatchNormalization()(enc)

    enc = layers.LeakyReLU(alpha=0.2)(enc)

    if drop_rate > 0.0:
        enc = layers.Dropout(drop_rate)(enc)

    return enc

def create_generator(latent_dim):
    init = keras.initializers.GlorotUniform() #RandomNormal(stddev = 0.03)

    #Structure
    latent_input = layers.Input(shape=latent_dim)

    x = layers.Dense(8 * 8 * 192//2, use_bias=False)(latent_input)
    x = layers.Reshape((8, 8, 192//2))(x)

    #----------Encoder
    drop_rate = 0.0

    x = enc_layer(x,  192//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #16x16   128 //2
    x = enc_layer(x,  256//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #32x32   192 //2
    x = enc_layer(x,  320//2, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #64x64   256 //2
    #x = enc_layer(x,  1024//4, kernel_size=(4,4), strides=(1,1), drop_rate=drop_rate, kernel_initializer=init)

    # 128, 192, 256 = t3! 102 looks very good!!!! , 7k and 64 batch, tain 1.5 1.25

    # 192, 256, 320  t4, 10k and 64 batch, tain 1.5 1.25

    # 192, 256, 320 new t5, 20k and 64 batch, tain 1.5 1.25

    # 130, 192, 256  t6, 30k and 64 batch, tain 1.5 1.25
    # 130, 192, 256  t7, 10k and 128 batch, tain 1.5 1.25


    #------- Activation-layer
    output = layers.Conv2D(1, kernel_size=(5,5), strides=1, padding='same', activation=activations.tanh)(x)

    #----------- Model
    g_model = keras.models.Model(inputs = latent_input, outputs=output, name="generator")
    return g_model
