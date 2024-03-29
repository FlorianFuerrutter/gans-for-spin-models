import sys, os.path
import numpy as np
import importlib
from dataclasses import dataclass, field


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

            if last_loaded_epoch_index < 0:
                states_epoch.append(np.zeros((3, image_size[0] * image_size[1] * image_size[2])))

            return np.array(states_epoch), last_loaded_epoch_index

        #generate spin data
        batch_size = 100

        latent_vectors = gan.sample_generator_input(batch_size, latent_dim)
        generated_images = (gan_model.generator(latent_vectors)).numpy()
       
        for i in range((images_count // batch_size) - 1):
            latent_vectors = gan.sample_generator_input(batch_size, latent_dim)
            t = (gan_model.generator(latent_vectors)).numpy()
            generated_images = np.concatenate((generated_images, t), axis=0)

        #clip to +-1
        images = (np.where(generated_images < 0.0, -1.0, 1.0)).astype(np.int8)

        if image_size is not None:
            images = np.reshape(images, (-1, image_size[0] * image_size[1] * image_size[2]))

        states_epoch.append(images)

    return np.array(states_epoch), last_loaded_epoch_index

#--------------------------------------------------------------------

def importConditionalGAN(gan_name):
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", gan_name)
       
    if ("conditional_gan" in sys.modules):
        #check if imported
        sys.path.pop() #remove last path, this only works if no other path changes were done
        sys.path.append(model_path)
        conditional_gan = importlib.reload(sys.modules["conditional_gan"])
    else:
        #else import it
        sys.path.append(model_path)
        import conditional_gan

    return conditional_gan

def generate_conditional_gan_data(TJs, gan_name="Spin_DC_GAN", epochs=range(20, 91, 10), images_count=1000, latent_dim=128, conditional_dim=1, image_size=(64, 64, 1), alt_path=False):
    conditional_gan = importConditionalGAN(gan_name)

    if gan_name == "Spin_StyleGAN2":
        enc_block_count = 5
        style_dim = 4096
        noise_image_res = 64
        gan_model = conditional_gan.conditional_gan(enc_block_count, latent_dim, conditional_dim, style_dim, image_size, noise_image_res)      
    else:
        gan_model = conditional_gan.conditional_gan(latent_dim, conditional_dim, image_size)

    if alt_path:
        gan_model.save_path = os.path.join(os.path.dirname(__file__), "..", "models", gan_name, "model-saves", "gan_")
    else:
        gan_model.save_path = os.path.join(model_data_path, gan_name,"c_gan", "gan_")

    last_loaded_epoch_index = -1
    states_epoch_tj = [] # (epochs, tjs, states)
    epochs          = np.array(epochs).astype(int)

    for epoch_index in range(epochs.shape[0]):
        epoch = epochs[epoch_index]
        
        #load weights of gan model 
        try:
            gan_model.load(epoch)
            last_loaded_epoch_index = epoch_index
        except:
            print("[generate_gan_data] Not loaded:", gan_name, ", epoch:", epoch)

            if last_loaded_epoch_index < 0:
                states_epoch_tj.append( np.zeros((TJs.shape[0], images_count, image_size[0] * image_size[1] * image_size[2]), dtype=np.int8) )

            return np.array(states_epoch_tj), last_loaded_epoch_index

        print("[generate_gan_data] Loaded:", gan_name, ", epoch:", epoch)

        #generate spin data
        batch_size = 128
        states_tj = [] # (tjs, states)

        for TJ in TJs:
            conditional_labels = np.ones((batch_size, conditional_dim)) * TJ

            if gan_name == "Spin_StyleGAN2":
                random_vectors, noise_images = conditional_gan.sample_generator_input(batch_size, enc_block_count, latent_dim, noise_image_res)
                latent_vectors = [np.concatenate([random_vector, conditional_labels], axis=1) for random_vector in random_vectors]

                generated_images = (gan_model.generator([latent_vectors, noise_images])).numpy()
            else:
                random_vectors = conditional_gan.sample_generator_input(batch_size, latent_dim)
                latent_vectors = np.concatenate([random_vectors, conditional_labels], axis=1)

                generated_images = (gan_model.generator(latent_vectors)).numpy()
       
            for i in range((images_count // batch_size) - 1):          
                if gan_name == "Spin_StyleGAN2":
                    random_vectors, noise_images = conditional_gan.sample_generator_input(batch_size, enc_block_count, latent_dim, noise_image_res)
                    latent_vectors = [np.concatenate([random_vector, conditional_labels], axis=1) for random_vector in random_vectors]

                    t = (gan_model.generator([latent_vectors, noise_images])).numpy()
                else:
                    random_vectors = conditional_gan.sample_generator_input(batch_size, latent_dim)
                    latent_vectors = np.concatenate([random_vectors, conditional_labels], axis=1)

                    t = (gan_model.generator(latent_vectors)).numpy()

                generated_images = np.concatenate((generated_images, t), axis=0)

            #clip to +-1
            images = (np.where(generated_images < 0.0, -1.0, 1.0)).astype(np.int8)
       
            images = np.reshape(images, (-1, image_size[0] * image_size[1] * image_size[2]))

            states_tj.append(images)

        states_epoch_tj.append(np.array(states_tj))

    return np.array(states_epoch_tj), last_loaded_epoch_index

#--------------------------------------------------------------------

def load_spin_observables(TJ, addpath):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train")
    #path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train", "64")

    file_path = os.path.join(path, addpath + "simulation_observ_TJ_{TJ}.npy".format(TJ=TJ))

    obser = np.transpose(np.load(file_path))
    #obser = np.transpose(np.loadtxt(file_path, skiprows=1, dtype=np.float32))

    energy = obser[0]
    m      = obser[1]
    mAbs   = obser[2]
    m2     = obser[3]
    mAbs3  = obser[4]
    m4     = obser[5]
    
    print("[load_spin_observables] Found data count:", energy.shape[0])
    return energy, m, mAbs, m2, mAbs3, m4

def load_spin_states(TJ, addpath):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train")
    file_path = os.path.join(path, addpath + "simulation_states_TJ_{TJ}.npy".format(TJ=TJ))

    states = np.load(file_path)

    print("[load_spin_states] Found data count:", states.shape[0])
    return states

#--------------------------------------------------------------------

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

    xi          : float = -1
    xi_err      : float = -1
    g_xi        : float = -1
    g_xi_err    : float = -1
    
@dataclass
class err_data:
    val : float
    err : float 
    std : float

@dataclass
class model_processed_data:
    model_name    : str = ""
    model_name_id : int = 0

    TJs : list[float] = field(default_factory=lambda : [])

    obs_dist     : float = -1
    obs_dist_std : float = -1
    obs_dist_min : float = -1
    obs_dist_max : float = -1

    energy   : list[err_data] = field(default_factory=lambda : [])
    mAbs     : list[err_data] = field(default_factory=lambda : [])
    magSusc  : list[err_data] = field(default_factory=lambda : [])
    binderCu : list[err_data] = field(default_factory=lambda : [])
    k3       : list[err_data] = field(default_factory=lambda : [])
    xi       : list[err_data] = field(default_factory=lambda : [])

    g_energy   : list[err_data] = field(default_factory=lambda : [])
    g_mAbs     : list[err_data] = field(default_factory=lambda : [])
    g_magSusc  : list[err_data] = field(default_factory=lambda : [])
    g_binderCu : list[err_data] = field(default_factory=lambda : [])
    g_k3       : list[err_data] = field(default_factory=lambda : [])
    g_xi       : list[err_data] = field(default_factory=lambda : [])

#--------------------------------------------------------------------
