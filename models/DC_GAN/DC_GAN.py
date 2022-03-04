import os
import numpy as np
import matplotlib.pyplot as plt

#Tensorflow
import tensorflow as tf
from tensorflow import keras
print("[INFO] Using tf version: " + tf.__version__ )
print("[INFO] Build with gpu support: " + format(tf.test.is_built_with_gpu_support()))

device_name = tf.test.gpu_device_name()  
if device_name != '/device:GPU:0':
   print("[WARNING] GPU device not found, using CPU for inference")
   print(" -> TensorFlow GPU support requires an assortment of drivers and libraries: https://www.tensorflow.org/install/gpu#software_requirements")
else:
    print('[INFO] Found GPU at: {}'.format(device_name))

#-------------------------------------
#-------------------------------------

import gan
import generator
import discriminator

#-------------------------------------
#-------------------------------------

def main() -> int:  
    #--------------
    #setup
    epochs     = 100
    latent_dim = 128

    image_size = (64, 64, 3)
    batch_size = 256
    
    #--------------
    #load data
    path     = os.path.join(os.path.dirname(__file__), "..", "..", "..", "img_align_celeba_part1")
    log_path = os.path.join(os.path.dirname(__file__), "..", "logs")

    #-----
       
    if 0:
        #if existing dataset, use that
        dataset = tf.data.experimental.load(path)
    elif 0:
        #create and store new dataset
        dataset = tf.keras.utils.image_dataset_from_directory(
                        path,
                        label_mode=None, 
                        image_size=(image_size[0], image_size[1]),
                        batch_size=batch_size,
                        smart_resize=True)
        dataset = dataset.map(lambda x: (x - 127.5) / 127.5)    
        tf.data.experimental.save(dataset, path) 

    #-----

    if 1:
        #load
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
                  
        #28x28 to 64x64 with padding, (64-28)/2=18
        x_train = np.stack((x_train,) * 3, axis=-1).astype("float32")
        images = tf.image.pad_to_bounding_box(x_train, 18, 18, 64, 64)  

        #convert to Dataset
        dataset = tf.data.Dataset.from_tensor_slices(images)
        dataset = dataset.map(lambda x: (x - 127.5) / 127.5)    
        dataset = dataset.batch(batch_size)

    if 0:    
        count = 1
        fig = plt.figure(figsize=(6, 6))
        for x in dataset:
            images = x.numpy()
            for image in images:
                image = (image + 1.0) /2.0

                plt.subplot(3, 3, count)
                plt.axis('off')
                plt.imshow(image)  

                count +=1
                if count > 9:
                    plt.savefig(path + "/sample.png", bbox_inches='tight')
                    plt.show()
                    exit(0)

    #--------------
    #define loss and optimizer

    def g_loss_fn(y_true, y_pred):
        # a logit,         (i.e, value in [-inf, inf] when from_logits=True ) 
        # or a probability (i.e, value in [0.,    1.] when from_logits=False)
        keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True, label_smoothing=0.0)
    def d_loss_fn(y_true, y_pred):
        # a logit,         (i.e, value in [-inf, inf] when from_logits=True ) 
        # or a probability (i.e, value in [0.,    1.] when from_logits=False)
        keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0.0)

    g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
   
    #--------------
    #create model

    g_model   = generator.create_generator(latent_dim)
    d_model   = discriminator.create_discriminator(image_size)

    g_model.summary()
    d_model.summary()

    k = keras.losses.BinaryCrossentropy()

    gan_model = gan.gan(d_model, g_model, latent_dim)
    gan_model.compile(d_optimizer, g_optimizer, k, k)

    #--------------
    #train
    gan_model.fit(dataset, epochs=epochs, callbacks=[gan.train_callback(latent_dim=latent_dim), keras.callbacks.TensorBoard(log_dir=log_path)])
 
    #save_path = './keras-saves/_gan_model_latest.ckpt'
    #try:
    #    gan_model.fit(
    #except KeyboardInterrupt:
        #needs implementation first!
        #gan_model.save(save_path)
        #print('Output saved to: "{}./*"'.format(save_path))

    return 0


if __name__ == '__main__':
    main()