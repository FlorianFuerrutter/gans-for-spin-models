import sys, os.path
import numpy as np
import importlib

#--------------------------------------------------------------------

model_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "model-data")

#--------------------------------------------------------------------

def importGAN(gan_name):
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", gan_name)
       
    if ("gan" in sys.modules):
        #check if imported
        sys.path.pop() #remove last path, this only works if no other path changes were done
        sys.path.append(model_path)
        gan = importlib.reload(sys.modules["gan"])
    else:
        #else import it
        sys.path.append(model_path)
        import gan

    return gan

def generate_gan_data(TJ, gan_name="Spin_DC_GAN", epochs=range(20, 91, 10), images_count=1000, latent_dim=128, image_size=(64, 64, 1)):
    gan = importGAN(gan_name)

    gan_model = gan.gan(latent_dim, image_size)
    gan_model.save_path = os.path.join(model_data_path, "gan_name","TJ_{TJ}".format(TJ=TJ), "gan_")

    states_epoch = []
    for epoch in epochs:

        #load weights gan model for tj
        gan_model.load(epoch)

        #generate spin data
        batch_size = 100

        latent_vectors = gan.sample_generator_input(batch_size, latent_dim)
        generated_images = (gan_model.generator(latent_vectors)).numpy()
       
        for i in range((images_count // batch_size) - 1):
            latent_vectors = gan.sample_generator_input(batch_size, latent_dim)
            t = (gan_model.generator(latent_vectors)).numpy()
            generated_images = np.concatenate((generated_images, t), axis=0)

        #clip to +-1
        images = np.where(generated_images < 0, -1, 1)
       
        if image_size is not None:
            images = np.reshape(images, (-1, image_size[0] * image_size[1] * image_size[2]))

        states_epoch.append(images)

    return np.array(states_epoch)

#--------------------------------------------------------------------

def load_spin_observables(TJ):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train")
    file_path = os.path.join(path, "simulation_observ_TJ_{TJ}.txt".format(TJ=TJ))

    obser = np.transpose(np.loadtxt(file_path, skiprows=1, dtype=np.float32))
    energy = obser[0]
    m2     = obser[1]
    
    print("found data count:", energy.shape[0])
    return energy, m2

#--------------------------------------------------------------------

from dataclasses import dataclass

@dataclass
class model_evaluation_data:

    #MC data
    mAbs   : np.ndarray = None
    energy : np.ndarray = None
   
    #GAN data
    g_states : np.ndarray = None
    g_mAbs   : np.ndarray = None
    g_energy : np.ndarray = None
      
    #metrics of best best_epoch
    best_epoch: int = -1

    mag_pol : float = -1
    mag_emd : float = -1
    eng_pol : float = -1
    eng_emd : float = -1

@dataclass
class err_data:
    val : float = 0
    err : float = 0

@dataclass
class model_processed_data:
    mAbs   : list[err_data]
    energy : list[err_data]

    g_mAbs   : list[err_data]
    g_energy : list[err_data]


#--------------------------------------------------------------------