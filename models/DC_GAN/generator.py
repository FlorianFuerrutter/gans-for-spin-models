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


    x = layers.Dense(8 * 8 * 128, use_bias=False)(latent_input)
    x = layers.Reshape((8, 8, 128))(x) #designed to match discriminator flatten 

    #----------Encoder
    x = enc_layer(x,  128, kernel_size=(4,4), strides=(2,2), drop_rate=0.0, kernel_initializer=init)
    x = enc_layer(x,  256, kernel_size=(4,4), strides=(2,2), drop_rate=0.0, kernel_initializer=init)
    x = enc_layer(x,  512, kernel_size=(4,4), strides=(2,2), drop_rate=0.0, kernel_initializer=init)

    #------- Activation-layer
    output = layers.Conv2D(3, kernel_size=(5,5), strides=1, padding='same', activation=activations.tanh)(x)

    #----------- Model
    g_model = keras.models.Model(inputs = latent_input, outputs=output, name="generator")
    return g_model