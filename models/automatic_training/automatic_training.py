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

def main() -> int:
    amplitude   = 0.7
    image_size = (64, 64, 1)

    batch_sizes = {"Spin_DC_GAN" : 64}
    latent_dims = {"Spin_DC_GAN" : 4096}
    conditional = {"Spin_DC_GAN" : 1}

    conditional_dim = 2

    #---------------------------
    model_names = np.array(["Spin_DC_GAN"])
    #TJs         = np.array([1.0, 1.8, 2.0, 2.2, 2.25, 2.3, 2.4, 2.6, 3.4])
    TJs = np.array([1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.3, 2.35, 2.4, 2.5, 2.6, 2.8, 3.0, 3.4])
    
    #TJs         = np.array([1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.26, 2.27, 2.3, 2.4, 2.5, 2.6, 3.0, 3.4,
    #                        2.28, 2.29, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37, 2.38, 2.39])

    epochs      = 1003
   
    plot_period = 2
    save_period = 2
  
    #---------------------------
    train_data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train") #, "64")

    for model_name in model_names:
        gan_module = import_gan_module(model_name)

        if conditional[model_name]:
            print("[train_conditional_model] model_name:", model_name, "--------------------------------")
            
            #load data 
            dataset = gan_module.load_conditional_spin_data(batch_sizes[model_name], image_size[0], train_data_path, TJs, amplitude=amplitude)

            # train to fixed epoch
            weights_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model-data", model_name, "c_gan", "gan_")

            gan_module.train_conditional_model(dataset, epochs, save_period, plot_period, latent_dims[model_name], conditional_dim, image_size, weights_path, plot_path=weights_path[:-5])
          
        else:
            for TJ in TJs:
                print("[train_model] model_name:", model_name, ", TJ:", TJ, "--------------------------------")

                #--------------
                #load data          
                file_name = "simulation_states_TJ_{TJ}.txt".format(TJ=TJ)
                print("[train_model] load_spin_data:", file_name)
                dataset = gan_module.load_spin_data(batch_sizes[model_name], image_size[0], train_data_path, name=file_name, amplitude=amplitude)
 
                # train to fixed epoch
                weights_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "model-data", model_name, "TJ_{TJ}".format(TJ=TJ), "gan_")

                gan_module.train_model(dataset, epochs, save_period, plot_period, latent_dims[model_name], image_size, weights_path, plot_path=weights_path[:-5])

    #---------------------------
    return 0

#-----------------------------------------------------------------

if __name__ == '__main__':
    main()