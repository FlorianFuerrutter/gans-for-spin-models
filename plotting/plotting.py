import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------
#-------------------------------------

def reduceIntoBin2(data):
    binnedSize = data.shape[0] // 2
    binnedData = np.zeros(binnedSize)
    
    for j in range(binnedSize):
        binnedData[j] = data[2*j] + data[2*j + 1]

    return binnedData * 0.5

def binningAnalysisSingle(data, printAlreadyConvergedWarning=False):
    #algo find max of errors -> use this binsize
    #if not converging -> max is last element then report error converging

    N = data.shape[0]
    maxBinningSteps = int(np.log2(N)) #so last binning has 2 elements, 1 element is useless for error

    mean       = np.zeros(maxBinningSteps)
    meanErrors = np.zeros(maxBinningSteps)

    #starting data
    binnedData    = data
    mean[0]       = np.mean(data)
    meanErrors[0] = np.std(data) / np.sqrt(N)

    #binning  up to maxBinningSteps
    for i in range(1, maxBinningSteps):
        #binsize = 2**i

        #binning step
        binnedData = reduceIntoBin2(binnedData)
        N = binnedData.shape[0] 

        #new error, mean
        mean[i]       = np.mean(binnedData)
        meanErrors[i] = np.std(binnedData) / np.sqrt(N)

    maxElement = np.argmax(meanErrors)
    if (maxElement+1) == maxBinningSteps: 
        info = "   (elements in bin of max error: " + str(data.shape[0] // 2**maxElement) + ")"
        print("binningAnalysisSingle: [NOT CONVERGED]  increase dataset," + info)
    if maxElement == 0 and printAlreadyConvergedWarning:
        info = "   (elements in bin of max error: " + str(data.shape[0] // 2**maxElement) + ")"
        print("binningAnalysisSingle: [Already CONVERGED]  first error is largest," + info)

    #print("max error at binstep=%d, binsize=%d" % (mi, 2**mi))  
    corrSize = 2**maxElement
    return mean[maxElement], meanErrors[maxElement], corrSize

#-------------------------------------
#-------------------------------------

import sys, os.path
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "Spin_DC_GAN")          
sys.path.append(model_path)
import gan

plot_path  = os.path.dirname(__file__) #+ "plot"
def savePdf(filename): 
  plt.savefig(plot_path + "/" + filename + '.pdf', bbox_inches='tight')

def generate_gan_data(TJ, images_count=1000):
    latent_dim = 128
    image_size = (64, 64, 1)

    gan_model = gan.gan(latent_dim, image_size)
    gan_model.save_path = os.path.join(os.path.dirname(__file__), "..", "data", "generated", "TJ_{TJ}".format(TJ=TJ), "gan_")

    states_epoch = []
    for epoch in range(20, 91, 10):

        #load weights gan model for tj
        gan_model.load(epoch)

        #generate spin data
        batch_size = 100

        latent_vectors = gan.sample_generator_input(batch_size, latent_dim)
        generated_images = (gan_model.generator(latent_vectors)).numpy()
       
        for i in range(images_count // batch_size - 1):
            latent_vectors = gan.sample_generator_input(batch_size, latent_dim)
            t = (gan_model.generator(latent_vectors)).numpy()
            generated_images = np.concatenate((generated_images, t), axis=0)

        #clip to +-1
        images = np.where(generated_images < 0, -1, 1)
       
        state = np.reshape(images, (-1, image_size[0] * image_size[1] * image_size[2]))
        states_epoch.append(state)
    return np.array(states_epoch)

def load_spin_data(TJ):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train")
    file_path = os.path.join(path, "simulation_observ_TJ_{TJ}.txt".format(TJ=TJ))
    obser = np.transpose(np.loadtxt(file_path, skiprows=1, dtype=np.float32))

    energy = obser[0]
    m2     = obser[1]
    
    print("found data count:", energy.shape[0])
    return energy, m2

#-------------------------------------
#-------------------------------------

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'font.size': 16,
    'mathtext.fontset': 'cm',
    'font.family': 'STIXGeneral',
    'axes.unicode_minus': True})

def plot_comparison(TJs, means, errs, g_means, g_errs):
    size=(12, 4.8*1.5)
    fig = plt.figure(figsize = size, constrained_layout = True) 

    plt.xlabel(r'$T/J$')                   
    plt.xticks(TJs)
    plt.ylabel(r"$m^2$")

    plt.plot(TJs, means, "--", color="tab:blue", alpha=0.5, linewidth=0.8)
    plt.errorbar(TJs, means, fmt='.', yerr=errs, label="Simulated", elinewidth=1, capsize=5, markersize=5)

    for i in range(len(g_means[0])):
        epoch = (i+2)*10

        g_mean_epochs = g_means[:, i]
        g_err_epochs  = g_errs[:, i]

        plt.errorbar(TJs, g_mean_epochs, fmt='.', yerr=g_err_epochs, label="DC-GAN epoch: %d" % epoch, elinewidth=1, capsize=5, markersize=5)

    plt.legend()
    savePdf("comp_v1")
    plt.savefig(plot_path + "/" + "comp_v1" + '.png', bbox_inches='tight')
    plt.show()

#-------------------------------------
#-------------------------------------

def do():
    means = []
    errs  = []
    g_means = []
    g_errs  = []

    TJs = [1.8, 2, 2.2, 2.5]
    for TJ in TJs:  #-> convert data into value
        
        #load MC spin data
        energy, m2 = load_spin_data(TJ)

        #compute values for MC values -> binning
        mean, err, corr = binningAnalysisSingle(m2)

        means.append(mean)
        errs.append(err)

        #load GAN spin data
        generated_states = generate_gan_data(TJ, images_count=1000)
        N = 64*64
        g_m2 = np.square(np.sum(generated_states, axis=2)) / N**2

        #compute values for GAN values -> binning
        g_data = np.transpose([binningAnalysisSingle(x) for x in g_m2])
        g_mean, g_err, g_corr = g_data[0], g_data[1], g_data[2]

        g_means.append(g_mean)
        g_errs.append(g_err)

    plot_comparison(np.array(TJs), np.array(means), np.array(errs), np.array(g_means), np.array(g_errs))
    return

#-------------------------------------
#-------------------------------------

def main() -> int:
    do()
    return 0

if __name__ == '__main__':
    main()