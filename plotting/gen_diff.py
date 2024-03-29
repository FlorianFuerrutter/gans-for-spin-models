import numpy as np
import tensorflow as tf
import data_visualization
import model_evaluation
import data_helper
import data_analysis
import os
from scipy.ndimage import uniform_filter1d
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
#plot_path = "F:/GAN - Plots"
plot_path = "C:/Users/Flo/Documents/Uni/Ba-Arbeit/ba thesis/img/plots" 

ck_path   = "F:/GAN - DC_CK"

def savePdf(filename): 
    plt.savefig(plot_path + "/" + filename + '.pdf', bbox_inches='tight')
def savePng(filename):
    plt.savefig(plot_path + "/" + filename + '.png', bbox_inches='tight')
def saveSvg(filename):
    plt.savefig(plot_path + "/" + filename + '.svg', bbox_inches='tight')

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

def getStates_gradient_DCGAN(T, gan_model, samples, conditional_dim, random_vectors, res):

    #------- do batches for memory ---------------------
    mini_batch_size = 128
    runs = (samples // mini_batch_size) + 1

    for i in range(runs):
        with tf.GradientTape() as tape: 
            tf_T = tf.convert_to_tensor(T,  dtype=tf.dtypes.float32)
            tf_random_vectors = tf.convert_to_tensor(random_vectors)

            tape.watch(tf_T)

            tf_conditional_labels = tf.ones((samples, conditional_dim)) * tf_T
            tf_latent_vectors = tf.concat([tf_random_vectors, tf_conditional_labels], -1)

            tf_slice_latent_vectors = tf_latent_vectors[i*mini_batch_size:(i+1)*mini_batch_size]
       
            out_tensors = gan_model.generator(tf_slice_latent_vectors)

            tf_grad = tape.gradient(out_tensors, tf_T) #automatic takes sum -> normalize needed!!
            tf_grad = np.abs( (tf_grad / (res * res * tf_slice_latent_vectors.shape[0])).numpy() )

        if i == 0:
            grad = tf_grad
        else:
            grad = np.append(grad, tf_grad)

    return np.nanmean(grad)

def calcGenGradient_DCGAN(T, conditional_gan, gan_model, samples, conditional_dim, latent_dim, res):
    print("T:", T)

    random_vectors = conditional_gan.sample_generator_input(samples, latent_dim)

    grad = getStates_gradient_DCGAN(T, gan_model, samples, conditional_dim, random_vectors, res)

    return grad

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
    d = np.mean( np.square(TdT_states - T_states) ) / dT
    d = np.mean( np.abs(TdT_states - T_states) ) / dT
    v = np.mean( np.abs(T_states) )

    return d, v

#--------------------------------------------------------------------

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

    clrs = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    reses  = [16, 32, 48, 64]
    epochs = [ 3, 21, 13, 26]
    dTs    = np.array([ 4, 3, 2, 1]) * 0.0032

    Fs = list()
    vs = list()
    v  = F = 0

    gen_new = 0

    #reses  = [64]
    #epochs = [26]
    #dTs    = np.array([1]) * 0.0032

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

            F = [calcGenGradient_DCGAN(T, conditional_gan, gan_model, samples, conditional_dim, latent_dim, res) for T in Ts]

            #data = [calcGenDiff_DCGAN(T, dT, conditional_gan, gan_model, samples, conditional_dim, latent_dim) for T in Ts]
            #F = [x[0] for x in data]
            #v = [x[1] for x in data]
         
            plot_path = "F:/GAN - Plots"
            np.save(plot_path + "/data/" + gan_name + "_L%d_GenDiff_Ts" % res, Ts)
            np.save(plot_path + "/data/" + gan_name + "_L%d_GenDiff_F"  % res, F)    
            #np.save(plot_path + "/data/" + gan_name + "_L%d_GenDiff_v"  % res, v)         
        else:
            try:
                plot_path = "F:/GAN - Plots"
                F = np.load(plot_path + "/data/" + gan_name + "_L%d_GenDiff_F.npy"  % res) 
                #v = np.load(plot_path + "/data/" + gan_name + "_L%d_GenDiff_v.npy"  % res) 
            except:
                continue

        Fs.append(F)
        vs.append(v)

    #-------------------------------------------
    size=(12, 4.1)
    fig = plt.figure(figsize = size, constrained_layout = True) 
    matplotlib.rcParams.update({'font.size': 18})

    plt.margins(0.03) 
    plt.xlabel(r'$T/J$')
    plt.ylabel(r"$\mathcal{F}_{\mathrm{GAN}}(T)$")
    if 0:
        plt.xlim((2.1,2.7))
        plt.ylim((0.7,3.2))
        plt.yscale('log')

    Tc = 1.0 * 2.0 / np.log(1.0 + np.sqrt(2.0))
    plt.axvline(Tc, color="gray", linestyle="--", linewidth=1.6)

    for k in range(len(Fs)):
        i = len(Fs) - 1 - k 

        F   = Fs[i]
        res = reses[i]
        clr = clrs[i]

        #--------------------------
        #legend
        te = r"$L$ = %d" % res
        args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color=clr, size="large")
        #plt.text(1.02, 0.98-0.12*k, te, args)

        #--------------------------
        #F = F / np.max(F)
        kernel_size = 10
        F_smooth = uniform_filter1d(F, kernel_size, mode="nearest")

        peak = Ts[np.argmax(F_smooth)]
        plt.axvline(peak, color=clr, linestyle="--", alpha=0.4)

        plt.plot(Ts, F, "--", alpha=0.3, linewidth=0.7, color=clr)
        plt.plot(Ts, F_smooth, label=r"$L$ = $%d$" % res, color=clr)

    plt.legend()

    #-------------------------------------------
    savePdf(gan_name + "_GenDiff")
    #savePng(gan_name + "_GenDiff")
    #saveSvg(gan_name + "_GenDiff")
    return

#--------------------------------------------------------------------

if __name__ == '__main__':
    main()

    plt.show()

#--------------------------------------------------------------------
