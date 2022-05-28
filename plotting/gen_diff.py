import numpy as np
import data_visualization
import model_evaluation
import data_helper
import data_analysis
import os
import matplotlib
import matplotlib.pyplot as plt
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

def getStates_DCGAN(T, gan_model, samples, conditional_dim, random_vectors):
    conditional_labels = np.ones((samples, conditional_dim)) * T   
    latent_vectors = np.concatenate([random_vectors, conditional_labels], axis=1)

    #------- do batches for memory ---------------------
    mini_batch_size = 128
    runs = (samples // mini_batch_size) + 1
    for i in range(runs):

        slice_latent_vectors = latent_vectors[i*mini_batch_size:(i+1)*mini_batch_size]
        generated_images = (gan_model.generator(slice_latent_vectors)).numpy()

        if i == 0:
            images = generated_images
        else:
            images = np.concatenate((images, generated_images), axis=0)

    return images

def calcGenDiff_DCGAN(T, dT, conditional_gan, gan_model, samples, conditional_dim, latent_dim):
    print("T:", T)

    random_vectors = conditional_gan.sample_generator_input(samples, latent_dim)

    T_states   = getStates_DCGAN(T     , gan_model, samples, conditional_dim, random_vectors)
    TdT_states = getStates_DCGAN(T + dT, gan_model, samples, conditional_dim, random_vectors)

    #d = np.mean( np.square(TdT_states - T_states) ) / dT
    d = np.mean(TdT_states - T_states) / dT

    return d

#--------------------------------------------------------------------

def main():
    gan_name = "Spin_DC_GAN"
    model_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "model-data")

    conditional_gan = importConditionalGAN(gan_name)

    latent_dim = 4096
    conditional_dim = 4
    image_size = (64, 64, 1)
    gan_model = conditional_gan.conditional_gan(latent_dim, conditional_dim, image_size)

    gan_model.save_path = os.path.join(model_data_path, gan_name,"ck", "gan_")
    gan_model.load(epoch=26)

    #-------------------------------------------
    Ts = np.linspace(1.0, 3.4, 750)
    dT = Ts[1] - Ts[0] 
    samples = 2**11

    F = [calcGenDiff_DCGAN(T, dT, conditional_gan, gan_model, samples, conditional_dim, latent_dim) for T in Ts]

    np.save(plot_path + "/" + gan_name + "_GenDiff_Ts", Ts)
    np.save(plot_path + "/" + gan_name + "_GenDiff_F", F)

    #-------------------------------------------
    size=(12, 5)
    fig = plt.figure(figsize = size, constrained_layout = True) 
    plt.xlabel(r"$T$")
    plt.ylabel(r"$d_{\mathrm{GAN}}(T)$")

    Tc = 1.0 * 2.0 / np.log(1.0 + np.sqrt(2.0))
    plt.axvline(Tc, color="gray", linestyle="--")

    plt.plot(Ts, F)

    savePdf(gan_name + "_GenDiff")
    savePng(gan_name + "_GenDiff")
    return

#--------------------------------------------------------------------

if __name__ == '__main__':
    main()

    plt.show()
#--------------------------------------------------------------------
