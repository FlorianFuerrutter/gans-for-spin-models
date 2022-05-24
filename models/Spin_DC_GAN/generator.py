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

def injection_layer(x, latent_input, res, filter_size, kernel_initializer, conditional_dim=1):

    inject = layers.Dense(filter_size//2, use_bias=False, kernel_initializer=kernel_initializer)(latent_input[:, -conditional_dim:])
    inject = layers.LeakyReLU(alpha=0.2)(inject)
    
    inject = layers.Dense(filter_size//2, kernel_initializer=kernel_initializer)(inject)
    inject = layers.LeakyReLU(alpha=0.2)(inject)
    inject = layers.Dropout(0.2)(inject)
 
    #------------------------------------------------------

    inject = layers.RepeatVector(res*res)(inject)
    inject = layers.Reshape((res, res, filter_size//2))(inject)

    inject = layers.Concatenate(axis=-1)([x, inject])
    inject = layers.Conv2D(filter_size, kernel_size=(3,3), strides=(1,1), padding="same", kernel_initializer=kernel_initializer)(inject)

    #------------------------------------------------------

    inject = layers.Add()([x, inject])
    inject = layers.LeakyReLU(alpha=0.2)(inject)

    return inject

def create_generator(latent_dim, conditional_dim=0, injection=True):
    init = keras.initializers.GlorotUniform() #RandomNormal(stddev = 0.03)

    scale = 1

    #Structure
    latent_input = layers.Input(shape=latent_dim)

    res = 8

    x = layers.Dense(res * res * 64//scale, use_bias=False)(latent_input) #128//2   
    x = layers.Reshape((res, res, 64//scale))(x)

    #----------Encoder
    drop_rate = 0.0
   
    x = enc_layer(x,  128//scale, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #16x16   128 //2
    if conditional_dim > 0 and injection:
        x = injection_layer(x, latent_input, 16, 128//scale, init, conditional_dim)

    x = enc_layer(x,  192//scale, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #32x32   192 //2   
    if conditional_dim > 0 and injection:
        x = injection_layer(x, latent_input, 32, 192//scale, init, conditional_dim)

    x = enc_layer(x,  256//scale, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init) #64x64   256 //2   
    if conditional_dim > 0 and injection:
        x = injection_layer(x, latent_input, 64, 256//scale, init, conditional_dim)

    #x = enc_layer(x,  320//4, kernel_size=(4,4), strides=(2,2), drop_rate=drop_rate, kernel_initializer=init)

    # 128, 192, 256 , epoch 102 looks very good!!!! , 7k and 64 batch, tain 1.5 1.25 (up to 132)
    # 192, 256, 320 , 20k and 64 batch, tain 1.5 1.25  half ok up to 114
    # 130, 192, 256 , 30k and 64 batch, tain 1.5 1.25, gets better,but bad, up to 27
    # 130, 192, 256 , 10k and 128 batch, tain 1.5 1.25 colabsd at 42 els seems ok

    # --------
    #gen: 130 192 256 // 2, batch=64, data=10k, tain 1.5 1.25
    #dis:  64 128 128 // 2

    #------- Activation-layer
    output = layers.Conv2D(1, kernel_size=(5,5), strides=1, padding='same', activation=activations.tanh)(x)

    #----------- Model
    g_model = keras.models.Model(inputs = latent_input, outputs=output, name="generator")
    return g_model
