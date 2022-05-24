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
plot_path = "F:/GAN - Animation"

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

def create_gaussian_linspace(Tmin, Tmax):
    Tc = 1.0 * 2.0 / np.log(1.0 + np.sqrt(2.0))
    x = np.array([Tc])

    def dist(T):
        p = 600.0 * np.exp( -np.square(T - Tc) / (2 * 0.2**2) ) + 300.0
        return 1 / p

    T = Tc
    while 1:
        T -= dist(T)

        if T < Tmin:
            x = np.append(x, Tmin)
            break
        
        x = np.append(x, T)

    T = Tc
    while 1:
        T += dist(T)

        if T > Tmax:
            x = np.append(x, Tmax)
            break
        
        x = np.append(x, T)

    return np.sort(x)

#--------------------------------------------------------------------

def create_conditional_image(img, T, Tmin, Tmax):
    fig = plt.figure(figsize=(10,10), constrained_layout=True, dpi=115) 
    plt.axis('off')
    plt.ylim( 0.0, 1.12)
    plt.xlim(-0.2, 1.20)

    def T_to_axis(T, xmin, xmax, Tmin, Tmax):
        k = (xmax - xmin) / (Tmax - Tmin)
        d = xmin - k * Tmin
        return T * k + d

    #paramters
    xmin = -0.1
    xmax = 1.1
    y      = 1.08
    height = 0.07

    #---------- Axis ----------
    plt.hlines(y, xmin, xmax)
    plt.vlines(xmin, y - height / 2.0, y + height / 2.0)
    plt.vlines(xmax, y - height / 2.0, y + height / 2.0)

    plt.text(xmin - 0.03, y, r'$%.1f$' % Tmin, horizontalalignment='right', verticalalignment="center", size="xx-large")
    plt.text(xmax + 0.03, y, r'$%.1f$' % Tmax, horizontalalignment='left',  verticalalignment="center", size="xx-large")

    Tc = 1.0 * 2.0 / np.log(1.0 + np.sqrt(2.0))
    Tc_x = T_to_axis(Tc, xmin, xmax, Tmin, Tmax)
    plt.vlines(Tc_x, y - height / 2.0, y + height / 2.0, color="gray", linestyle="--")
    plt.text(Tc_x + 0.01, y - 0.025, r'$T_c$', horizontalalignment='left', verticalalignment="top", size="xx-large")

    #---------- Cond label ----------
    label_x = T_to_axis(T, xmin, xmax, Tmin, Tmax)

    plt.plot(label_x, y, 'ro', ms=15, mfc='r')
    plt.text(label_x, y + 0.045, r'$T$ = $%.2f$' % T, horizontalalignment='center', verticalalignment="bottom", color="red", size="xx-large")

    #---------- Image ----------
    extent = (0, 1, 0, 1) 
    plt.imshow(img, origin="lower", extent=extent, vmin=-1.0, vmax=1.0)

    return fig

def create_conditional_states_styleGAN(epoch, Ts, latent_vector, noise_image):
    gan_name = "Spin_StyleGAN2"
    model_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "model-data")

    conditional_gan = importConditionalGAN(gan_name)

    latent_dim = 4096
    conditional_dim = 8
    enc_block_count = 5
    style_dim = 4096
    noise_image_res = 64
    image_size = (64, 64, 1)
    gan_model = conditional_gan.conditional_gan(enc_block_count, latent_dim, conditional_dim, style_dim, image_size, noise_image_res)

    gan_model.save_path = os.path.join(model_data_path, gan_name,"ck", "gan_")
    gan_model.load(epoch)

    #---------------------
    batch_size = Ts.size
    conditional_labels = np.array([np.ones(conditional_dim) * T for T in Ts])

    if 0: #random
        random_vectors, noise_images = conditional_gan.sample_generator_input(batch_size, enc_block_count, latent_dim, noise_image_res)
        latent_vectors = [np.concatenate([random_vector, conditional_labels], axis=1) for random_vector in random_vectors]
    else: #fixed latent
        noise_images   = [noise_image for T in Ts]
        latent_vectors = [latent_vector for T in Ts] * enc_block_count
        latent_vectors = [np.concatenate([latent_vector, conditional_labels], axis=1) for latent_vector in latent_vectors]

    generated_images = (gan_model.generator([latent_vectors, noise_images])).numpy()

    #clip to +-1
    images = (np.where(generated_images < 0.0, -1.0, 1.0)).astype(np.int8)   
    
    return images

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

def create_conditional_gif(epoch, Ts):
    enc_block_count = 5
    batch_size = Ts.size
    latent_dim = 4096
    noise_image_res = 64

    #----- create fixed latent -----
    latent_vector = np.random.normal(0, 1, size=(latent_dim))

    #----- create fixed direct noise inputs -----
    noise_image   = []
    for i in range(enc_block_count):
        noise_image.append(np.random.normal(0, 1, size=(noise_image_res, noise_image_res, 1)))

    #----- gen states -----
    #states = create_conditional_states_styleGAN(epoch, Ts, latent_vector, noise_image)
    states = create_conditional_states_DCGAN(epoch, Ts, latent_vector)

    #----- create images -----
    for i in range(Ts.size):
        T     = Ts[i]
        state = states[i]

        fig = create_conditional_image(state, T, 1.0, 3.4)
        savePng("c_Img_%d" % i)
        plt.close(fig)

#--------------------------------------------------------------------

def main():
    matplotlib.use('Agg')

    #Ts = np.linspace(1.0, 3.4, 10)
    Ts = create_gaussian_linspace(1.0, 3.4) #1024 points

    create_conditional_gif(epoch=26, Ts=Ts)

    return

#--------------------------------------------------------------------

if __name__ == '__main__':
    main()

#--------------------------------------------------------------------