import sys, os.path
import numpy as np
import importlib
import tensorflow as tf

#-----------------------------------------------------------------

def import_gan_module(gan_name=""):
    model_path = os.path.join(os.path.dirname(__file__), "..", gan_name)

    sys.path.append(model_path)
    gan_module = importlib.import_module(gan_name)

    return gan_module

#-----------------------------------------------------------------

def load_spin_data(batch_size, res, path, name, amplitude=0.9, load_dataset=False):

    if load_dataset:
        dataset = tf.data.experimental.load(path)
    else:
        #create new dataset 
        file_path = os.path.join(path, name)
        states = np.loadtxt(file_path, skiprows=1, dtype=np.float32)
        states = np.reshape(states, ( -1, res, res, 1))
        print("[load_spin_data] Found states:", states.shape[0])

        #scale (+-)1 to (+-)amplitude
        states = states * amplitude

        dataset = tf.data.Dataset.from_tensor_slices(states)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        #tf.data.experimental.save(dataset, path) 

    return dataset

#-----------------------------------------------------------------

def main() -> int:
    image_size = (64, 64, 1)

    batch_sizes = {"Spin_DC_GAN" : 64}
    latent_dims = {"Spin_DC_GAN" : 256}

    #---------------------------
    model_names = np.array(["Spin_DC_GAN"])
    TJs         = np.array([1.0, 1.8, 2.0, 2.2, 2.25, 2.3, 2.4, 2.6, 3.4])

    epochs      = 81
    amplitude   = 0.7
   
    plot_period = 3 #2 * epochs -> not needed, so never
    save_period = 3 #10

    #---------------------------
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train")

    for model_name in model_names:
        gan_module = import_gan_module(model_name)

        for TJ in TJs:
            print("[train_model] model_name:", model_name, ", TJ:", TJ, "--------------------------------")

            #--------------
            #load data          
            file_name = "simulation_states_TJ_{TJ}.txt".format(TJ=TJ)
            print("[train_model] load_spin_data:", file_name)
            dataset = load_spin_data(batch_sizes[model_name], image_size[0], path, name=file_name, amplitude=amplitude)

            # train to fixed epoch
            weights_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model-data", model_name, "TJ_{TJ}".format(TJ=TJ), "gan_")

            gan_module.train_model(dataset, epochs, save_period, plot_period, latent_dims[model_name], image_size, weights_path, plot_path=weights_path[:-5])

    #---------------------------
    return 0

#-----------------------------------------------------------------

if __name__ == '__main__':
    main()