import numpy as np
import data_helper as dh
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from scipy.stats import gaussian_kde 
import model_evaluation as me
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

#def plot_performance_evaluation_

#--------------------------------------------------------------------

def plot_performance_evaluation_hist(med_objs : dh.model_evaluation_data, use_energy_not_m=False):
    cnt = len(med_objs)

    cols = 3
    rows = int(np.ceil(cnt / cols))

    #--------------------------------
    size=(13, 3.4 * rows) #TODO scale according to count
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(rows, cols, figure=fig)

    if use_energy_not_m:
        ticks = np.linspace(me.range_eng[0], me.range_eng[1], 6)
    else:
        ticks = np.linspace(me.range_m[0], me.range_m[1], 5)
    empty_labels = ["" for x in ticks]

    #--------------------------------
    axs = np.array([])
    for y in range(rows):
        for x in range(cols):
            if y * cols + x >= cnt:
                continue
            axs = np.append(axs, fig.add_subplot(gs[y,x]))
      
    #---------------------------
    clr_sim = "tab:blue"
    clr_gan = "tab:orange"

    #legend
    plt.sca(axs[2 if cnt>2 else cnt-1])
    if med_objs[0].model_name_id == 0:
        args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color=clr_sim, size="large")
        plt.text(1.03, 0.96, r"Simulated", args)

    args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color=clr_gan, size="large")
    plt.text(1.03, 0.96-0.13*(med_objs[0].model_name_id+1), r"{m}".format(m=med_objs[0].model_name), args)

    #--------------------------------
    for iy in range(rows):
        for ix in range(cols):
            i = iy * cols + ix
            if i >= cnt:
                continue

            med = med_objs[i]

            #--------------------------------
            plt.sca(axs[i])
            plt.margins(0.03) 

            te = r"$T/J$ = {t}".format(t=med.T)
            args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color="black")#, bbox=box)
            plt.text(0.08, 0.98 , te, args)
            
            #--------------------------------
            if use_energy_not_m:
                te  = r"$\mathrm{POL}$: %2.1f" % (med.eng_pol*100.0) + "\n"
                te += r"$\mathrm{EMD}$: %2.1f" % (med.eng_emd)
            else:
                te  = r"$\mathrm{POL}$: %2.1f" % (med.m_pol*100.0) + "\n"
                te += r"$\mathrm{EMD}$: %2.1f" % (med.m_emd)

            args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color="black")#, bbox=box)
            plt.text(0.65, 0.971 , te, args)

            #--------------------------------
            if ix == 0:
                plt.ylabel(r"Probability density")
            if iy == rows-1:
                if use_energy_not_m:
                    plt.xlabel(r"$E$")
                else:
                    plt.xlabel(r"$m$")
                plt.xticks(ticks)
            else:
                plt.xticks(ticks, empty_labels)

            #--------------------------------
            #Simulation
            data   = med.m           
            bins   = me.bin_size_m
            b_range = me.range_m
            label  = "Simulation m"

            if use_energy_not_m:
                data  = med.energy              
                bins  = me.bin_size_eng
                b_range = me.range_eng
                label = "Simulation E"

            plt.hist(data, bins=bins, range=b_range, density=True, label=label, alpha=0.5, color=clr_sim)

            if 0:
                density_sim = gaussian_kde(data, bw_method=lambda x: 0.25)
                xs = np.linspace(b_range[0], b_range[1], 500)
                plt.plot(xs, density_sim(xs), color=clr_sim)

            
            #--------------------------------
            #GAN
            data  = med.g_m           
            bins  = me.bin_size_m
            b_range = me.range_m
            label = "GAN m"

            if use_energy_not_m:
                data  = med.g_energy           
                bins  = me.bin_size_eng
                b_range = me.range_eng
                label = "GAN E"

            plt.hist(data, bins=bins, range=b_range, density=True, label=label, alpha=0.5, color=clr_gan)

            if 0: 
                density_gan = gaussian_kde(med.g_m, bw_method=lambda x: 0.25)
                xs = np.linspace(b_range[0], b_range[1], 500)
                plt.plot(xs, density_gan(xs), color=clr_gan)

            #--------------------------------
            #plt.legend()
 
    t = "energy_" if use_energy_not_m else "m_"
    savePdf("plot_performance_evaluation_hist_" + t + med_objs[-1].model_name)
    savePng("plot_performance_evaluation_hist_" + t + med_objs[-1].model_name)
    #plt.show()
    return

def plot_performance_evaluation_phase(med_objs : dh.model_evaluation_data):
    cnt = len(med_objs)

    cols = 3
    rows = int(np.ceil(cnt / cols))

    #--------------------------------
    size=(12, 3.4 * rows) #TODO scale according to count
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(rows, cols, figure=fig)

    x_ticks = np.linspace(me.range_m[0], me.range_m[1], 5)
    x_empty_labels = ["" for x in x_ticks]

    y_ticks = np.linspace(me.range_eng[0], me.range_eng[1], 6)
    y_empty_labels = ["" for y in y_ticks]

    alpha_min = 0.2
    alpha_max = 1
    #--------------------------------
    cmap_sim = plt.get_cmap("Blues")
    cmap_sim_lin        = cmap_sim(np.arange(cmap_sim.N))
    cmap_sim_lin[:,-1]  = np.linspace(alpha_min, alpha_max, cmap_sim.N)
    cmap_sim_lin[0, -1] = 0
    cmap_sim = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap_sim_lin)

    #--------------------------------
    cmap_gan = plt.get_cmap("Oranges")   
    cmap_gan_lin       = cmap_gan(np.arange(cmap_gan.N))
    cmap_gan_lin[:,-1] = np.linspace(alpha_min, alpha_max, cmap_gan.N)
    cmap_gan_lin[0, -1] = 0
    cmap_gan = matplotlib.colors.LinearSegmentedColormap.from_list("", cmap_gan_lin)

    #--------------------------------
    axs = np.array([])
    for y in range(rows):
        for x in range(cols):
            if y * cols + x >= cnt:
                continue
            axs = np.append(axs, fig.add_subplot(gs[y,x]))
      
    #--------------------------------
    for iy in range(rows):
        for ix in range(cols):
            i = iy * cols + ix
            if i >= cnt:
                continue

            med = med_objs[i]

            #--------------------------------
            plt.sca(axs[i])
            plt.margins(0.03) 
            plt.gca().set_facecolor((0.2, 0.2, 0.2))

            off = 1.05
            plt.xlim((me.range_m[0]*off, me.range_m[1]*off))
            plt.ylim((me.range_eng[0]*off, me.range_eng[1]*off))

            #--------------------------------
            te = r"$T/J$ = {t}".format(t=med.T)
            args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color="white")#, bbox=box)
            plt.text(0.08, 0.98 , te, args)

            #--------------------------------
            te  = r"$\mathrm{POL}$: %2.1f" % (med.phase_pol*100.0) + "\n"
            #te += r"$\mathrm{EMD}$: $%2.1f$" % (med.m_emd)
            args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color="white")#, bbox=box)
            plt.text(0.65, 0.971 , te, args)

            #--------------------------------
            if ix == 0:
                plt.ylabel(r"$E$")
                plt.yticks(y_ticks)
            else:
                plt.yticks(y_ticks, y_empty_labels)

            if iy == rows-1:
                plt.xlabel(r"$m$")
                plt.xticks(x_ticks)
            else:
                plt.xticks(x_ticks, x_empty_labels)

            #--------------------------------
            H_spin, H_gan, xedges, yedges = me.create_hist2D(med.m, med.energy, med.g_m, med.g_energy, bin_scale=0.3)
            X, Y = np.meshgrid(xedges, yedges)
                    
            #plt.pcolormesh(X, Y, H_spin, rasterized=True, cmap=cmap_sim, label="Simulation")
            plt.pcolormesh(X, Y, H_gan, rasterized=True, cmap=cmap_gan, label="GAN")

            dx = np.abs(X[0,0] - X[0,1])
            dy = np.abs(Y[0,0] - Y[1,0])

            plt.contour(X[:-1, :-1] + dx/2., Y[:-1, :-1] + dy/2., H_spin, cmap="bone", levels=2, linewidths=2) #, label="Simulation")
            #plt.contour(X[:-1, :-1] + dx/2., Y[:-1, :-1] + dy/2., H_gan, cmap=cmap_gan, levels=2, linewidths=2.5) #, label="GAN")

            #plt.contourf(X[:-1, :-1] + dx/2., Y[:-1, :-1] + dy/2., H_spin, cmap=cmap_sim) #, label="Simulation")
            #plt.contourf(X[:-1, :-1] + dx/2., Y[:-1, :-1] + dy/2., H_gan, cmap=cmap_gan) #, label="GAN")
     
            #also plot somehow the evolution with epochs !!

    savePdf("plot_performance_evaluation_phase_" + med_objs[-1].model_name)
    savePng("plot_performance_evaluation_phase_" + med_objs[-1].model_name)
    #plt.show()
    return

#--------------------------------------------------------------------

def plot_performance_evaluation_observables(TJs, mpd : dh.model_processed_data):
    Tc = 1.0 * 2.0 / np.log(1.0 + np.sqrt(2.0))

    #---------------------------
    size=(13, 4.8*1.6)
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(2, 2, figure=fig)
    axs = np.array([fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
                    fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])])
        
    #---------------------------
    title = [r"$\langle |m|\rangle$",  r"$\langle E\rangle$",
             r"$\chi$",                r"$\kappa_3$"] #r"$U_2$"

    #labels = [("%0.1f" % x) for x in TJs]   
    #labels.append(r"$T_c$")
    #ticks = TJs #np.append(TJs, Tc) 
    ticks  = [1.0, 1.8, 2.0, 2.2, 2.4, 2.6, 3.4]
    labels = [("%0.1f" % x) for x in ticks]  
    
    empty_labels = ["" for x in ticks]

    mc_data_list  = [mpd.mAbs  , mpd.energy,   mpd.magSusc,   mpd.k3] #mpd.binderCu]
    gan_data_list = [mpd.g_mAbs, mpd.g_energy, mpd.g_magSusc, mpd.g_k3] #mpd.g_binderCu] 

    #---------------------------
    clr_sim = "tab:blue"
    clr_gan = "tab:orange"

    #legend
    plt.sca(axs[1])
    if mpd.model_name_id == 0:
        args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color=clr_sim, size="large")
        plt.text(1.03, 0.96, r"Simulated", args)

    args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color=clr_gan, size="large")
    plt.text(1.03, 0.96-0.20*(mpd.model_name_id+1),      r"{m}".format(m=mpd.model_name), args)
    plt.text(1.03, 0.96-0.20*(mpd.model_name_id+1)-0.10, r"OOP: %.2f" % mpd.obs_dist, args)

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
            mc_data  = mc_data_list[i]
            gan_data = gan_data_list[i]

            mean, err     = [x.val for x in mc_data],  [x.err for x in mc_data]
            g_mean, g_err = [x.val for x in gan_data], [x.err for x in gan_data]

            plt.plot(TJs, mean, "--", color=clr_sim, alpha=0.5, linewidth=0.8)
            plt.errorbar(TJs, mean, fmt='.', yerr=err, label="Simulated", elinewidth=1, capsize=5, markersize=5, color=clr_sim)

            plt.errorbar(TJs, g_mean, fmt='.', yerr=g_err, label="GAN", elinewidth=1, capsize=5, markersize=5, color=clr_gan)
            #plt.legend()

            if i==3 and 0:
                plt.sca(axs[i].twinx())

                mc_data  = mpd.binderCu
                gan_data = mpd.g_binderCu

                mean, err     = [x.val for x in mc_data],  [x.err for x in mc_data]
                g_mean, g_err = [x.val for x in gan_data], [x.err for x in gan_data]

                plt.plot(TJs, mean, "--", color=clr_sim, alpha=0.5, linewidth=0.8)
                plt.errorbar(TJs, mean, fmt='.', yerr=err, label="Simulated", elinewidth=1, capsize=5, markersize=5, color=clr_sim)

                plt.errorbar(TJs, g_mean, fmt='.', yerr=g_err, label="GAN", elinewidth=1, capsize=5, markersize=5, color=clr_gan)


    savePdf("plot_performance_evaluation_observables_" + mpd.model_name)
    savePng("plot_performance_evaluation_observables_" + mpd.model_name)
    #plt.show()
    return