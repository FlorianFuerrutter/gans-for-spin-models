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

def generate_gan_data(TJ, gan_name="Spin_DC_GAN", epochs=range(20, 91, 10), images_count=1000, latent_dim=128, image_size=(64, 64, 1), alt_path=False):
    gan = importGAN(gan_name)

    gan_model = gan.gan(latent_dim, image_size)
    if alt_path:
        gan_model.save_path = os.path.join(os.path.dirname(__file__), "..", "models", gan_name, "model-saves", "gan_")
    else:
        gan_model.save_path = os.path.join(model_data_path, gan_name,"TJ_{TJ}".format(TJ=TJ), "gan_")

    last_loaded_epoch_index = -1
    states_epoch = []
    epochs       = np.array(epochs).astype(int)

    for epoch_index in range(epochs.shape[0]):
        epoch = epochs[epoch_index]

        #load weights gan model for tj
        try:
            gan_model.load(epoch)
            last_loaded_epoch_index = epoch_index
        except:
            print("[generate_gan_data] Not loaded:", gan_name, ", epoch:", epoch)
            states_epoch.append(np.zeros((images_count, image_size[0] * image_size[1] * image_size[2])))
            continue

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

    return np.array(states_epoch), last_loaded_epoch_index

#--------------------------------------------------------------------

def load_spin_observables(TJ):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train")
    file_path = os.path.join(path, "simulation_observ_TJ_{TJ}.txt".format(TJ=TJ))

    obser = np.transpose(np.loadtxt(file_path, skiprows=1, dtype=np.float32))
    energy = obser[0]
    m      = obser[1]
    mAbs   = obser[2]
    m2     = obser[3]
    mAbs3  = obser[4]
    m4     = obser[5]
    
    print("[load_spin_observables] Found data count:", energy.shape[0])
    return energy, m, mAbs, m2, mAbs3, m4

#--------------------------------------------------------------------

from dataclasses import dataclass, field

@dataclass
class model_evaluation_data:
    T : float = -1
    N : int   = -1
    model_name    : str = ""
    model_name_id : int = 0

    #MC data
    energy : np.ndarray = None
    m      : np.ndarray = None
    mAbs   : np.ndarray = None
    m2     : np.ndarray = None
    mAbs3  : np.ndarray = None
    m4     : np.ndarray = None
     
    #GAN data
    g_states : np.ndarray = None
    g_energy : np.ndarray = None
    g_m      : np.ndarray = None
    g_mAbs   : np.ndarray = None
    g_m2     : np.ndarray = None
    g_mAbs3  : np.ndarray = None
    g_m4     : np.ndarray = None
      
    #metrics of best best_epoch
    best_epoch: int = -1

    m_pol   : float = -1
    m_emd   : float = -1
    mAbs_pol : float = -1
    mAbs_emd : float = -1
    eng_pol : float = -1
    eng_emd : float = -1

    phase_pol : float = -1
    obs_dist  : float = -1

@dataclass
class err_data:
    val : float = 0
    err : float = 0

@dataclass
class model_processed_data:
    model_name    : str = ""
    model_name_id : int = 0
    obs_dist      : float = -1

    energy   : list[err_data] = field(default_factory=lambda : [])
    mAbs     : list[err_data] = field(default_factory=lambda : [])
    magSusc  : list[err_data] = field(default_factory=lambda : [])
    binderCu : list[err_data] = field(default_factory=lambda : [])
    k3       : list[err_data] = field(default_factory=lambda : [])

    g_energy   : list[err_data] = field(default_factory=lambda : [])
    g_mAbs     : list[err_data] = field(default_factory=lambda : [])
    g_magSusc  : list[err_data] = field(default_factory=lambda : [])
    g_binderCu : list[err_data] = field(default_factory=lambda : [])
    g_k3       : list[err_data] = field(default_factory=lambda : [])

#--------------------------------------------------------------------