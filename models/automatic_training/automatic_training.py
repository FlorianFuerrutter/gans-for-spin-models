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

def load_spin_data(batch_size, res, path, name, amplitude=0.9):
    #create and store new dataset 
    file_path = os.path.join(path, name)
    states = np.loadtxt(file_path, skiprows=1, dtype=np.float32)
    states = np.reshape(states, ( -1, res, res, 1))
    print("[load_spin_data] Found states:", states.shape[0])

    #scale (+-)1 to (+-)amplitude
    states = states * amplitude

    dataset = tf.data.Dataset.from_tensor_slices(states)
    dataset = dataset.batch(batch_size)

    return dataset

#-----------------------------------------------------------------

def main() -> int:
    image_size = (64, 64, 1)

    batch_sizes = {"Spin_DC_GAN" : 128}
    latent_dims = {"Spin_DC_GAN" : 128}

    #---------------------------
    model_names = np.array(["Spin_DC_GAN"])
    TJs         = np.array([1.0, 1.8, 2.0, 2.2, 2.4, 2.6, 3.4])
    TJs = np.array([1.8])

    plot_period = 1
    save_period = 1

    epochs      = 101
    a           = 0.9

    #---------------------------
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train")

    for model_name in model_names:
        gan_module = import_gan_module(model_name)

        for TJ in TJs:

            #--------------
            #load data
            file_name = "simulation_states_TJ_{TJ}.txt".format(TJ=TJ)
            dataset = load_spin_data(batch_sizes[model_name], image_size[0], path, name=file_name, amplitude=a)

            # train to fixed epoch
            weights_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model-data", model_name, "TJ_{TJ}".format(TJ=TJ), "gan_")

            gan_module.train_model(dataset, epochs, save_period, plot_period, latent_dims[model_name], image_size, weights_path)

    #---------------------------
    return 0

#-----------------------------------------------------------------

if __name__ == '__main__':
    main()