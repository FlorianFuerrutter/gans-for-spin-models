import numpy as np
import data_helper as dh
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

#--------------------------------------------------------------------

import os
plot_path  = os.path.dirname(__file__) #+ "plot"
def savePdf(filename): 
    plt.savefig(plot_path + "/" + filename + '.pdf', bbox_inches='tight')
def savePng(filename):
    plt.savefig(plot_path + "/" + filename + '.png', bbox_inches='tight')

#--------------------------------------------------------------------

def plot_performance_evaluation(TJs, mpd : dh.model_processed_data):
    Tc = 1.0 * 2.0 / np.log(1.0 + np.sqrt(2.0))

    #---------------------------
    size=(12, 4.8*1.6)
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(2, 2, figure=fig)
    axs = np.array([fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
                    fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])])
        
    #---------------------------
    title = [r"$\langle |m|\rangle$",  r"$\langle E\rangle$",
             r"$\chi$",                r"$U_2$"]

    labels = [("%0.1f" % x) for x in TJs]
    labels.append(r"$T_c$")
    ticks  = np.append(TJs, Tc) 
    empty_labels = ["" for x in ticks]

    mc_data_list  = [mpd.mAbs  , mpd.energy,   list[dh.err_data], list[dh.err_data]] 
    gan_data_list = [mpd.g_mAbs, mpd.g_energy, list[dh.err_data], list[dh.err_data]] 

    #---------------------------
    for iy in range(2):
        for ix in range(2):
            i = iy * 2 + ix

            #---------------------------
            plt.sca(axs[i])
            plt.margins(0.03) 
            plt.axvline(Tc, color="gray", linestyle="--")
            plt.ylabel(title[i])

            if iy > 0:
                plt.xlabel(r'$T/J$')
                plt.xticks(ticks, labels)
            else:
                plt.xticks(ticks, empty_labels)

            #---------------------------
            #TODO ADD THE OTHER VARS
            if i > 0:
                continue

            mc_data  = mc_data_list[i]
            gan_data = gan_data_list[i]

            mean, err     = [x.val for x in mc_data],  [x.err for x in mc_data]
            g_mean, g_err = [x.val for x in gan_data], [x.err for x in gan_data]

            plt.plot(TJs, mean, "--", color="tab:blue", alpha=0.5, linewidth=0.8)
            plt.errorbar(TJs, mean, fmt='.', yerr=err, label="Simulated", elinewidth=1, capsize=5, markersize=5)

            plt.errorbar(TJs, g_mean, fmt='.', yerr=g_err, label="GAN", elinewidth=1, capsize=5, markersize=5)
            plt.legend()

    #savePdf("comp_v1")
    #savePng("comp_v1")
    plt.show()
    return