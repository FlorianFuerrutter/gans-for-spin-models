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

def plot_metrics_history(epochs, last_loaded_epoch_index, model_name, m_pol, mAbs_pol, eng_pol, m_emd, mAbs_emd, eng_emd, phase_pol, obs_dist):

    epochs = epochs[:last_loaded_epoch_index+1]
    m_pol = m_pol[:last_loaded_epoch_index+1]
    mAbs_pol = mAbs_pol[:last_loaded_epoch_index+1]
    eng_pol = eng_pol[:last_loaded_epoch_index+1]
    m_emd = m_emd[:last_loaded_epoch_index+1]
    mAbs_emd = mAbs_emd[:last_loaded_epoch_index+1]
    eng_emd = eng_emd[:last_loaded_epoch_index+1]
    phase_pol = phase_pol[:last_loaded_epoch_index+1]
    obs_dist = obs_dist[:last_loaded_epoch_index+1]

    #---------------------------
    size=(13, 4.8*1.8)
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(3, 1, figure=fig)
    #axs = np.array([fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]), fig.add_subplot(gs[3])])
    axs = np.array([fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]) ])

    ticks  = np.arange(0, epochs[-1], 50 if epochs[-1]<=250 else 100)
    labels = [("%d" % x) for x in ticks]    
    empty_labels = ["" for x in ticks]

    #---------------------------
    plt.sca(axs[0])
    plt.margins(0.03)
    plt.ylabel("%OL")
    plt.xticks(ticks, empty_labels)

    plt.plot(epochs, m_pol, "-", label="m_pol")
    plt.plot(epochs, mAbs_pol, "-", label="mAbs_pol")
    plt.plot(epochs, eng_pol, "-", label="eng_pol")
    plt.plot(epochs, phase_pol, "-", label="phase_pol")
    plt.legend()

    #---------------------------
    plt.sca(axs[1])
    plt.margins(0.03)
    plt.ylabel("EMD")
    plt.yscale('log')
    plt.xticks(ticks, empty_labels)
    #plt.ylim([0, 5])

    plt.plot(epochs, m_emd, "-", label="m_emd")
    plt.plot(epochs, mAbs_emd, "-", label="mAbs_emd")
    plt.plot(epochs, eng_emd, "-", label="eng_emd")
    plt.legend()

    #---------------------------
    plt.sca(axs[2])
    plt.margins(0.03)
    plt.ylabel("Obs distance")
    plt.xlabel("Epoch")
    plt.xticks(ticks, labels)
    plt.yscale('log') 
    #plt.ylim([0, 1.5])

    plt.plot(epochs, obs_dist, "-", label="obs_dist")
    plt.legend()

    #---------------------------
    #plt.sca(axs[3])
    #plt.margins(0.03)
    #plt.ylabel("Obs distance")
    #plt.xlabel("Epoch")
    #plt.ylim([1, 10])

    #plt.plot(epochs, obs_dist, "-", label="obs_dist")
    #plt.legend()
  
    #---------------------------
    #plt.show()
    savePdf("plot_metrics_history_" + model_name)
    savePng("plot_metrics_history_" + model_name)
    #exit(0)
    return

def plot_metrics_history_conditional(epochs, last_loaded_epoch_index, model_name, obs_dist):
    epochs = epochs[:last_loaded_epoch_index+1]

    #---------------------------
    size=(13, 4.8*1.4)
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(2, 1, figure=fig)
    axs = np.array([fig.add_subplot(gs[0]), fig.add_subplot(gs[1])])

    ticks  = np.arange(0, epochs[-1], 50 if epochs[-1]<=250 else 100)
    labels = [("%d" % x) for x in ticks]    
    empty_labels = ["" for x in ticks]

    #---------------------------
    plt.sca(axs[0])
    plt.margins(0.03)
    plt.ylabel("Obs distance")
    plt.xlabel("Epoch")
    plt.xticks(ticks, labels)

    plt.plot(epochs, obs_dist, "-", label="obs_dist")

    #---------------------------
    plt.sca(axs[1])
    plt.margins(0.03)
    plt.ylabel("Obs distance")
    plt.xlabel("Epoch")
    plt.xticks(ticks, labels)
    plt.yscale('log') 

    plt.plot(epochs, obs_dist, "-", label="obs_dist")

    #---------------------------
    savePdf("plot_metrics_history_conditional_" + model_name)
    savePng("plot_metrics_history_conditional_" + model_name)
    return


#--------------------------------------------------------------------

def plot_correlation_fit(x, y, fit_func, best_vals):
    size=(12, 4.8*1.0)
    fig = plt.figure(figsize = size, constrained_layout = True) 
    plt.xlabel(r"$r$")
    plt.ylabel(r"$G_c(r)$")

    #data
    plt.plot(x, y, ".", ms=16)

    #fit
    off = 0.04 * ( np.max(x) - np.min(x) ) 
    x_fit = np.linspace(np.min(x) - off, np.max(x) + off, num = len(x) * 30 + 1, endpoint = True) #x[0] - off
    y_fit = fit_func(best_vals, x_fit) 
    plt.plot(x_fit, y_fit, "--")

    args = dict(horizontalalignment='left',verticalalignment='top',transform = plt.gca().transAxes)
    xkor, ykor, space = 1.02, 0.98, 0.12 
    plt.text(xkor, ykor - 0*space, r"$a$  = %.3f" % best_vals[0], args)
    plt.text(xkor, ykor - 1*space, r"$\xi$  = %.3f" % best_vals[1], args)

    #plt.show()
    return

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
    cmap_sim = plt.get_cmap("Blues") #bones
    cmap_sim_lin        = cmap_sim(np.arange(cmap_sim.N))
    cmap_sim_lin[:,-1]  = np.linspace(alpha_min*2, alpha_max, cmap_sim.N)
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

            plt.contour(X[:-1, :-1] + dx/2., Y[:-1, :-1] + dy/2., H_spin, cmap=cmap_sim, levels=2, linewidths=2) #, label="Simulation") "bone"
            #plt.contour(X[:-1, :-1] + dx/2., Y[:-1, :-1] + dy/2., H_gan, cmap=cmap_gan, levels=2, linewidths=2.5) #, label="GAN")

            #plt.contourf(X[:-1, :-1] + dx/2., Y[:-1, :-1] + dy/2., H_spin, cmap=cmap_sim) #, label="Simulation")
            #plt.contourf(X[:-1, :-1] + dx/2., Y[:-1, :-1] + dy/2., H_gan, cmap=cmap_gan) #, label="GAN")

    savePdf("plot_performance_evaluation_phase_" + med_objs[-1].model_name)
    savePng("plot_performance_evaluation_phase_" + med_objs[-1].model_name)
    #plt.show()
    return

#--------------------------------------------------------------------

def plot_performance_evaluation_observables(TJs, mpd : dh.model_processed_data, mpd_interpolate : dh.model_processed_data = None):
    Tc = 1.0 * 2.0 / np.log(1.0 + np.sqrt(2.0))

    #---------------------------
    size=(13, 4.8*1.9)
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(3, 2, figure=fig)
    axs = np.array([fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
                    fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]),
                    fig.add_subplot(gs[2,0]), fig.add_subplot(gs[2,1])])
        
    #---------------------------
    title = [r"$\langle |m|\rangle$",  r"$\langle E\rangle$",
             r"$\chi$",                r"$\kappa_3$",
             r"$U_2$",                r"$\xi$"]

    #labels = [("%0.1f" % x) for x in TJs]   
    #labels.append(r"$T_c$")
    #ticks = TJs #np.append(TJs, Tc) 
    ticks  = [1.0, 1.8, 2.2, 2.6, 3.4]
    #ticks  = [1.0, 1.8, 2.0, 2.2, 2.4, 2.6, 3.4]
    labels = [("%0.1f" % x) for x in ticks]  
    
    empty_labels = ["" for x in ticks]

    mc_data_list  = [mpd.mAbs  , mpd.energy,   mpd.magSusc,   mpd.k3, mpd.binderCu, mpd.xi]
    gan_data_list = [mpd.g_mAbs, mpd.g_energy, mpd.g_magSusc, mpd.g_k3, mpd.g_binderCu, mpd.g_xi] 

    if mpd_interpolate != None:
        gan_interpolate_data_list = [mpd_interpolate.g_mAbs, mpd_interpolate.g_energy, mpd_interpolate.g_magSusc, mpd_interpolate.g_k3, mpd_interpolate.g_binderCu, mpd_interpolate.g_xi] 

    #---------------------------
    clr_sim = "tab:blue"
    clr_gan = "tab:orange"
    clr_interpolate = "tab:green"

    #legend
    plt.sca(axs[1])
    if mpd.model_name_id == 0:
        args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color=clr_sim, size="large")
        plt.text(1.03, 0.96, r"Simulated", args)

    args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color=clr_gan, size="large")
    plt.text(1.03, 0.96-0.20*(mpd.model_name_id+1),      r"{m}".format(m=mpd.model_name), args)
    plt.text(1.03, 0.96-0.20*(mpd.model_name_id+1)-0.10, r"OOP: $(%.1f \pm %0.1f)$" % (mpd.obs_dist, mpd.obs_dist_std), args)

    #---------------------------
    for iy in range(3):
        for ix in range(2):
            i = iy * 2 + ix

            #---------------------------
            plt.sca(axs[i])
            plt.margins(0.03) 
            plt.axvline(Tc, color="gray", linestyle="--")
            plt.ylabel(title[i])

            if iy > 1:
                plt.xlabel(r'$T/J$')
                plt.xticks(ticks, labels)
            else:
                plt.xticks(ticks, empty_labels)

            #---------------------------
            mc_data  = mc_data_list[i]
            gan_data = gan_data_list[i]

            mean, err, std       = [x.val for x in mc_data], [x.err for x in mc_data], [x.std for x in mc_data]
            g_mean, g_err, g_std = [x.val for x in gan_data], [x.err for x in gan_data], [x.std for x in gan_data]

            plt.plot(TJs, mean, "--", color=clr_sim, alpha=0.5, linewidth=0.8)
            plt.errorbar(TJs, mean, fmt='.', yerr=err, label="Simulated", elinewidth=1, capsize=5, markersize=5, color=clr_sim)
            plt.errorbar(TJs, mean, fmt='.', yerr=std, label="Simulated_std", elinewidth=1, capsize=2, markersize=5, color=clr_sim)

            plt.errorbar(TJs, g_mean, fmt='.', yerr=g_err, label="GAN", elinewidth=1, capsize=5, markersize=5, color=clr_gan)
            plt.errorbar(TJs, g_mean, fmt='.', yerr=g_std, label="GAN_std", elinewidth=1, capsize=2, markersize=5, color=clr_gan)

            if mpd_interpolate != None:
                gan_data = gan_interpolate_data_list[i]
                g_mean, g_err, g_std = [x.val for x in gan_data], [x.err for x in gan_data], [x.std for x in gan_data]

                plt.errorbar(mpd_interpolate.TJs, g_mean, fmt='.', yerr=g_err, label="GAN_interpolate", elinewidth=1, capsize=5, markersize=5, color=clr_interpolate, alpha=0.8)
                plt.errorbar(mpd_interpolate.TJs, g_mean, fmt='.', yerr=g_std, label="GAN_interpolate_std", elinewidth=1, capsize=2, markersize=5, color=clr_interpolate, alpha=0.8)

            if 0: #plot with 2 axis
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