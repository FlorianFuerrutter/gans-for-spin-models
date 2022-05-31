import numpy as np
import data_visualization
import model_evaluation
import data_helper
import data_analysis
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'font.size': 20,
    'mathtext.fontset': 'cm',
    'font.family': 'STIXGeneral',
    'axes.unicode_minus': True})

#plot_path = os.path.dirname(__file__)
plot_path = "F:/GAN - Plots"
ck_path   = "F:/GAN - DC_CK"

def savePdf(filename): 
    plt.savefig(plot_path + "/" + filename + '.pdf', bbox_inches='tight')
def savePng(filename):
    plt.savefig(plot_path + "/" + filename + '.png', bbox_inches='tight')

#--------------------------------------------------------------------

def importConditionalGAN(gan_name):
    import importlib, sys, os.path
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

#--------------------------------------------------------------------

def getStates_DCGAN(T, conditional_gan, gan_model, samples, conditional_dim, latent_dim):
    conditional_labels = np.ones((samples, conditional_dim)) * T
    random_vectors = conditional_gan.sample_generator_input(samples, latent_dim)
    latent_vectors = np.concatenate([random_vectors, conditional_labels], axis=1)

    #------- do batches for memory ---------------------
    mini_batch_size = 128
    runs = (samples // mini_batch_size) + 1
    for i in range(runs):

        slice_latent_vectors = latent_vectors[i*mini_batch_size:(i+1)*mini_batch_size]
        generated_images = (gan_model.generator(slice_latent_vectors)).numpy()

        generated_images = (np.where(generated_images < 0.0, -1.0, 1.0)).astype(np.int8)

        if i == 0:
            images = generated_images
        else:
            images = np.concatenate((images, generated_images), axis=0)
   
    return images

def getStates_StyleGAN(T, conditional_gan, gan_model, samples, conditional_dim, latent_dim, enc_block_count, noise_image_res):
    conditional_labels = np.ones((samples, conditional_dim)) * T
    random_vectors, noise_images = conditional_gan.sample_generator_input(samples, enc_block_count, latent_dim, noise_image_res)
    latent_vectors = [np.concatenate([random_vector, conditional_labels], axis=1) for random_vector in random_vectors]
   
    #------- do batches for memory ---------------------
    mini_batch_size = 128
    runs = (samples // mini_batch_size) + 1
    for i in range(runs):

        slice_latent_vectors = latent_vectors[i*mini_batch_size:(i+1)*mini_batch_size]
        slice_noise_images   = noise_images[i*mini_batch_size:(i+1)*mini_batch_size]

        generated_images = (gan_model.generator([slice_latent_vectors, slice_noise_images])).numpy()

        generated_images = (np.where(generated_images < 0.0, -1.0, 1.0)).astype(np.int8)

        if i == 0:
            images = generated_images
        else:
            images = np.concatenate((images, generated_images), axis=0)
   
    return images

#--------------------------------------------------------------------

def main():



    return

#--------------------------------------------------------------------

if __name__ == '__main__':
    main()

    plt.show()
#--------------------------------------------------------------------


