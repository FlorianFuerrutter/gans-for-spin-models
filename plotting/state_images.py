import numpy as np
import data_visualization as dv
import model_evaluation as me
import data_helper as dh
import data_analysis as da
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'font.size': 24,
    'mathtext.fontset': 'cm',
    'font.family': 'STIXGeneral',
    'axes.unicode_minus': True})

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

plot_path = "C:/Users/Flo/Documents/Uni/Ba-Arbeit/ba thesis/img/pic" 

def plotImages(name, rows, cols, imgs, global_cols, titles=None):

    size = (12,4.1)
    wspace = None

    if rows==1:
        size = (12,1.9) if titles!=None else (12,1.5)
        #wspace = 0.08

    fig = plt.figure(figsize=size, constrained_layout=False, dpi=180) 
    gs0 = plt.GridSpec(1, global_cols, figure=fig, wspace=wspace)
    
    axxs = list()
    for k in range(global_cols):
        gs_tmp = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[k], wspace=0.01, hspace=0.02)

        axs_tmp = np.array([])
        for x in range(cols):
            for y in range(rows):
                axs_tmp = np.append(axs_tmp, fig.add_subplot(gs_tmp[y,x]))    

        if titles != None:
            ax_ghost = fig.add_subplot(gs_tmp[:])
            ax_ghost.axis('off')
            ax_ghost.set_title(titles[k], fontdict={'fontsize':21})

        axxs.append(axs_tmp)

    j=0
    for axs in axxs:
        for x in range(cols):
            for y in range(rows):
                i = cols * y + x
                ax=axs[i]
                plt.sca(ax)
                plt.axis('off')
                plt.ylim(0, 1)
                plt.xlim(0, 1)

                extent = (0, 1, 0, 1) 
                plt.imshow(imgs[j], origin="lower", extent=extent, vmin=-1.0, vmax=1.0)
                j+=1
    plt.tight_layout()
       
    plt.savefig(plot_path + "/" + name + '.pdf', bbox_inches='tight')
    plt.savefig(plot_path + "/" + name + '.png', bbox_inches='tight')
    return

#--------------------------------------------------------------------

def create_conditional_states_DCGAN(epoch, Ts, latent_vector):
    gan_name = "Spin_DC_GAN"
    model_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "model-data")

    conditional_gan = importConditionalGAN(gan_name)

    latent_dim = 4096
    conditional_dim = 4
    image_size = (64, 64, 1)
    gan_model = conditional_gan.conditional_gan(latent_dim, conditional_dim, image_size)

    gan_model.save_path = os.path.join(model_data_path, gan_name,"ck", "gan_")
    gan_model.load(epoch)

    #---------------------
    batch_size = Ts.size
    conditional_labels = np.array([np.ones(conditional_dim) * T for T in Ts])

    if 0: #random
        random_vectors = conditional_gan.sample_generator_input(batch_size, latent_dim)
        latent_vectors = np.concatenate([random_vectors, conditional_labels], axis=1)
    else: #fixed latent 
        vecs = np.array([latent_vector for T in Ts])
        latent_vectors = np.concatenate([vecs, conditional_labels], axis=1)

    #------- do batches for memory ---------------------
    mini_batch_size = 200
    runs = (latent_vectors.shape[0] // mini_batch_size) + 1
    for i in range(runs):

        slice_latent_vectors = latent_vectors[i*mini_batch_size:(i+1)*mini_batch_size]
        generated_images = (gan_model.generator(slice_latent_vectors)).numpy()

        #clip to +-1
        generated_images = (np.where(generated_images < 0.0, -1.0, 1.0)).astype(np.int8) 

        if i == 0:
            images = generated_images
        else:
            images = np.concatenate((images, generated_images), axis=0)

    return images

def plotDCGAN_Sample(name, use_title=True):

    #temps = np.linspace(1.8, 3.0, 9)

    temps = np.array([1.2, 1.7, 2.1, 2.2, 2.25, 2.3, 2.4, 2.8, 3.4])

    titles = [r"$T={T}$".format(T=x) for x in temps] if use_title else None

    rows = 1
    cols = 1

   #------------------------
    Ts = np.array([])
    for t in temps:
        Ts = np.append(Ts, [t] * rows * cols)

    epoch = 26
    latent_dim = 4096
    latent_vector = np.random.normal(0, 1, size=(latent_dim))

    imgs = create_conditional_states_DCGAN(epoch, Ts, latent_vector)

    #------------------------
    plotImages(name, rows, cols, imgs, temps.size, titles)
    return

#--------------------------------------------------------------------

def plotTrainDataSetSample():
    Ts = np.array([1.8, 2.3, 3.4])

    imgs = list()

    rows = 2
    cols = 2

    for TJ in Ts:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "train")
        file_path = os.path.join(path, "simulation_states_TJ_{TJ}.npy".format(TJ=TJ))
        states = np.load(file_path)
        states = np.reshape(states, (-1, 64, 64))

        for x in states[np.random.choice(states.shape[0], rows*cols, replace=False)]:
            imgs.append(x)
    
    plotImages("train_set", rows, cols, imgs, Ts.size)
    return

#--------------------------------------------------------------------

if __name__ == '__main__':
   
    plotDCGAN_Sample("spin_gan_sample_1", 1)
    plotDCGAN_Sample("spin_gan_sample_2", 0)
    plotDCGAN_Sample("spin_gan_sample_3", 0)
    plotDCGAN_Sample("spin_gan_sample_4", 0)

    plotTrainDataSetSample()

    plt.show()

#--------------------------------------------------------------------