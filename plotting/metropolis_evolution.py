import numpy as np
import data_visualization as dv
import model_evaluation as me
import data_helper as dh
import data_analysis as da
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# 20 for pdf and 40 for presentation??
matplotlib.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': 'cmr10',
    'font.size': 20,
    'mathtext.fontset': 'cm',
    'font.family': 'STIXGeneral',
    'axes.unicode_minus': True})

plot_path    = "F:/GAN - MC Animation"

def savePdf(filename): 
    plt.savefig(plot_path + "/" + filename + '.pdf', bbox_inches='tight')
def savePng(filename):
    plt.savefig(plot_path + "/" + filename + '.png', bbox_inches='tight')
def saveSvg(filename):
    plt.savefig(plot_path + "/" + filename + '.svg', bbox_inches='tight')

#--------------------------------------------------------------------

def load_spin_states(TJ):
    path = os.path.join(os.path.dirname(__file__), "..", "data", "train", "thermal")
    file_path = os.path.join(path, "simulation_states_TJ_{TJ}.npy".format(TJ=TJ))

    states = np.load(file_path)

    print("[load_spin_states] Found data count:", states.shape[0])
    return states

def plot_states(states, T, maxImg):

    for i in range(states.shape[0]):
        img = states[i]

        if i > maxImg:
            return

        fig = plt.figure(figsize=(10,10), constrained_layout=True, dpi=115) 
        plt.axis('off')
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        #---------- Image ----------
        extent = (0, 1, 0, 1) 
        plt.imshow(img, origin="lower", extent=extent, vmin=-1.0, vmax=1.0)

        savePng("/{TJ}/".format(TJ=T) + "img_%d" % i)
        plt.close(fig)


def main():
    maxImg = 450

    Ts = np.array([1.1, 2.27, 3.4])

    #-----------------------------
    matplotlib.use('Agg') #else mem leak
   
    for T in Ts:
        print("T:", T)

        states = load_spin_states(T)
        states = np.reshape(states, (-1, 64, 64))

        plot_states(states, T, maxImg)

    return

#--------------------------------------------------------------------

if __name__ == '__main__':
    main()

    plt.show()

#--------------------------------------------------------------------

