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
    mini_batch_size = 200
    runs = (samples // mini_batch_size) + 1
    for i in range(runs):

        slice_latent_vectors = latent_vectors[i*mini_batch_size:(i+1)*mini_batch_size]
        generated_images = (gan_model.generator(slice_latent_vectors)).numpy()

        if i == 0:
            images = generated_images
        else:
            images = np.concatenate((images, generated_images), axis=0)

    return images

def getDisConfidence(T, T_states, gan_model, samples, res):
    w = h = res
    c = 1

    conditional_channel = np.ones(samples) * T
    conditional_channel = np.repeat(conditional_channel, repeats=[h * w])
    conditional_channel = np.reshape(conditional_channel, (-1, h, w, c))

    generated_conditional_images = np.concatenate([T_states, conditional_channel], axis=-1)

    T_conf, T_pred = gan_model.discriminator(generated_conditional_images)
    T_conf = 1 - T_conf.numpy()
    T_pred = T_pred.numpy()

    return T_conf

def calcGanFidelity_DCGAN(T, dT, conditional_gan, gan_model, samples, conditional_dim, latent_dim, res):

    print("T:", T)

    if 0: #dT in DIS
        T_states = getStates_DCGAN(T, conditional_gan, gan_model, samples, conditional_dim, latent_dim)

        T_conf   = getDisConfidence(T     , T_states, gan_model, samples, res)
        TdT_conf = getDisConfidence(T + dT, T_states, gan_model, samples, res)

        ganFidelity = np.mean(T_conf - TdT_conf) / dT

    else: #dT in GEN
        T_states   = getStates_DCGAN(T     , conditional_gan, gan_model, samples, conditional_dim, latent_dim)
        TdT_states = getStates_DCGAN(T + dT, conditional_gan, gan_model, samples, conditional_dim, latent_dim)

        T_conf   = getDisConfidence(T, T_states  , gan_model, samples, res)
        TdT_conf = getDisConfidence(T, TdT_states, gan_model, samples, res)

        ganFidelity = np.mean(T_conf - TdT_conf) / dT

    return ganFidelity

def load_ck(epoch, res, latent_dim, conditional_dim, conditional_gan):
    image_size = (res, res, 1)
    gan_model = conditional_gan.conditional_gan(latent_dim, conditional_dim, image_size)

    gan_model.save_path = os.path.join(ck_path, "L%d" % res, "gan_")
    gan_model.load(epoch=epoch)

    return gan_model

#--------------------------------------------------------------------

def main():
    gan_name = "Spin_DC_GAN"
    model_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "model-data")

    conditional_gan = importConditionalGAN(gan_name)

    latent_dim = 4096
    conditional_dim = 4
    image_size = (64, 64, 1)
    #gan_model = conditional_gan.conditional_gan(latent_dim, conditional_dim, image_size)
    #gan_model.save_path = os.path.join(model_data_path, gan_name,"ck", "gan_")
    #gan_model.load(epoch=26)

    clrs = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    reses  = [16, 32, 48, 64]
    epochs = [ 3, 21, 13, 26]
    dTs    = np.array([ 4, 3, 2, 1]) * 0.0032

    Fs = list()
    gen_new = 0

    #-------------------------------------------
    Ts = np.linspace(1.0, 3.4, 750)
    samples = 2**11

    for i in range(len(epochs)):
        res   = reses[i]
        epoch = epochs[i]
        dT    = dTs[i]

        print("process L%d" % res)

        if gen_new:
            try:
                gan_model = load_ck(epoch, res, latent_dim, conditional_dim, conditional_gan)
            except:
                continue

            F = [calcGanFidelity_DCGAN(T, dT, conditional_gan, gan_model, samples, conditional_dim, latent_dim, res) for T in Ts]

            np.save(plot_path + "/data/" + gan_name + "_L%d_GanFidelity_Ts" % res, Ts)
            np.save(plot_path + "/data/" + gan_name + "_L%d_GanFidelity_F"  % res, F)          
        else:
            try:
                F = np.load(plot_path + "/data/" + gan_name + "_L%d_GanFidelity_F.npy"  % res) 
            except:
                continue

        Fs.append(F)

    #-------------------------------------------
    size=(12, 5)
    fig = plt.figure(figsize = size, constrained_layout = True) 
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\mathcal{F}_{\mathrm{GAN}}(T)$")
    if 0:
        plt.xlim((2.1,2.7))
        plt.ylim((0.7,3.2))
        plt.yscale('log')

    Tc = 1.0 * 2.0 / np.log(1.0 + np.sqrt(2.0))
    plt.axvline(Tc, color="gray", linestyle="--")

    for i in range(len(Fs)):
        F   = Fs[i]
        res = reses[i]
        clr = clrs[i]
        
        #F = F / np.max(F)
        kernel_size = 20
        F_smooth = uniform_filter1d(F, kernel_size, mode="nearest")

        #peak = Ts[np.argmax(F)]
        #plt.axvline(peak, color=clr, linestyle="--")

        plt.plot(Ts, F, "--", alpha=0.3, linewidth=0.7, color=clr)
        plt.plot(Ts, F_smooth, label="L%d" % res, color=clr)
        #plt.plot(Ts, F, label="L%d" % res, color=clr)

    plt.legend()

    #-------------------------------------------
    savePdf(gan_name + "_GanFidelity")
    savePng(gan_name + "_GanFidelity")
    return

#--------------------------------------------------------------------

if __name__ == '__main__':
    main()

    plt.show()
#--------------------------------------------------------------------
