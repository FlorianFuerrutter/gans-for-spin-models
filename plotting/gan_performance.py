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

plot_path    = "F:/GAN - PerformancePlots"
dc_ck_path   = "F:/GAN - DC_CK"

mc_data_path = "F:/GAN - PerformancePlots/data MC"
dc_data_path = "F:/GAN - PerformancePlots/data DcGan"


def savePdf(filename): 
    plt.savefig(plot_path + "/" + filename + '.pdf', bbox_inches='tight')
def savePng(filename):
    plt.savefig(plot_path + "/" + filename + '.png', bbox_inches='tight')
def saveSvg(filename):
    plt.savefig(plot_path + "/" + filename + '.svg', bbox_inches='tight')

def save3(name):
    savePdf(name)
    savePng(name)
    saveSvg(name)

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

def load_ck(epoch, res, latent_dim, conditional_dim, conditional_gan, ck_path, injection=True):
    image_size = (res, res, 1)
    gan_model = conditional_gan.conditional_gan(latent_dim, conditional_dim, image_size, injection)

    if not injection:
        gan_model.save_path = os.path.join(ck_path, "L%d_noInj" % res, "gan_")
    else:
        gan_model.save_path = os.path.join(ck_path, "L%d" % res, "gan_")

    gan_model.load(epoch=epoch)

    return gan_model

#--------------------------------------------------------------------

def getStates_DCGAN(T, conditional_gan, gan_model, samples, conditional_dim, latent_dim):
    conditional_labels = np.ones((samples, conditional_dim)) * T
    random_vectors = conditional_gan.sample_generator_input(samples, latent_dim)
    latent_vectors = np.concatenate([random_vectors, conditional_labels], axis=1)

    #------- do batches for memory ---------------------
    mini_batch_size = 128
    runs = (samples // mini_batch_size) + 1
    for i in range(runs):

        slice_latent_vectors = latent_vectors[i*mini_batch_size:(i+1)*mini_batch_size]
        generated_images = (gan_model.generator(slice_latent_vectors)).numpy()

        generated_images = (np.where(generated_images < 0.0, -1.0, 1.0)).astype(np.int8)

        if i == 0:
            images = generated_images
        else:
            images = np.concatenate((images, generated_images), axis=0)
   
    return images

def getStates_StyleGAN(T, conditional_gan, gan_model, samples, conditional_dim, latent_dim, enc_block_count, noise_image_res):
    conditional_labels = np.ones((samples, conditional_dim)) * T
    random_vectors, noise_images = conditional_gan.sample_generator_input(samples, enc_block_count, latent_dim, noise_image_res)
    latent_vectors = [np.concatenate([random_vector, conditional_labels], axis=1) for random_vector in random_vectors]
   
    #------- do batches for memory ---------------------
    mini_batch_size = 128
    runs = (samples // mini_batch_size) + 1
    for i in range(runs):

        slice_latent_vectors = latent_vectors[i*mini_batch_size:(i+1)*mini_batch_size]
        slice_noise_images   = noise_images[i*mini_batch_size:(i+1)*mini_batch_size]

        generated_images = (gan_model.generator([slice_latent_vectors, slice_noise_images])).numpy()

        generated_images = (np.where(generated_images < 0.0, -1.0, 1.0)).astype(np.int8)

        if i == 0:
            images = generated_images
        else:
            images = np.concatenate((images, generated_images), axis=0)
   
    return images

#--------------------------------------------------------------------

def getMC_Observables(Ts, N, gen_new=False):
    addpath = ""
   
    #------------------------
    if gen_new:
        print("Generate getMC_Observables")

        if 1:
            e = list()
            e_rr = list()
            mAbs = list()
            mAbs_err = list()
            magSusc = list()
            magSusc_err = list()
            binderCu = list()
            binderCu_err = list()
            k3 = list()
            k3_err = list()
            xi = list()
            xi_err = list()

        for T in Ts:
            #load MC data (tj, bins)
            t_energy, t_m, t_mAbs, t_m2, t_mAbs3, t_m4 = dh.load_spin_observables(T, addpath)

            data_energy, data_mAbs, data_magSusc, data_binderCu, data_k3 = me.perform_observable_calculation(t_energy, t_m, t_mAbs, t_m2, t_mAbs3, t_m4, N, T)

            states = dh.load_spin_states(T, addpath)
            data_xi, data_xi_err = da.calc_spin_spin_correlation(states, N)

            #append
            if 1:
                e.append(data_energy.val)
                e_rr.append(data_energy.std)
                mAbs.append(data_mAbs.val)
                mAbs_err.append(data_mAbs.std)
                magSusc.append(data_magSusc.val)
                magSusc_err.append(data_magSusc.std)
                binderCu.append(data_binderCu.val)
                binderCu_err.append(data_binderCu.std)
                k3.append(data_k3.val)
                k3_err.append(data_k3.std)
                xi.append(data_xi)
                xi_err.append(data_xi_err)

        np.save(mc_data_path + "/Ts.npy", Ts)
        np.save(mc_data_path + "/e.npy", e)
        np.save(mc_data_path + "/e_rr.npy", e_rr)
        np.save(mc_data_path + "/mAbs.npy", mAbs)
        np.save(mc_data_path + "/mAbs_err.npy", mAbs_err)
        np.save(mc_data_path + "/magSusc.npy", magSusc)
        np.save(mc_data_path + "/magSusc_err.npy", magSusc_err)
        np.save(mc_data_path + "/binderCu.npy", binderCu)
        np.save(mc_data_path + "/binderCu_err.npy", binderCu_err)
        np.save(mc_data_path + "/k3.npy", k3)
        np.save(mc_data_path + "/k3_err.npy", k3_err)
        np.save(mc_data_path + "/xi.npy", xi)
        np.save(mc_data_path + "/xi_err.npy", xi_err)
                
    #------------------------
    else:
        print("Load getMC_Observables")
        Ts = np.load(mc_data_path + "/Ts.npy") 
        e = np.load(mc_data_path + "/e.npy")
        e_rr = np.load(mc_data_path + "/e_rr.npy")
        mAbs = np.load(mc_data_path + "/mAbs.npy")
        mAbs_err = np.load(mc_data_path + "/mAbs_err.npy")
        magSusc = np.load(mc_data_path + "/magSusc.npy")
        magSusc_err = np.load(mc_data_path + "/magSusc_err.npy")
        binderCu = np.load(mc_data_path + "/binderCu.npy")
        binderCu_err = np.load(mc_data_path + "/binderCu_err.npy")
        k3 = np.load(mc_data_path + "/k3.npy")
        k3_err = np.load(mc_data_path + "/k3_err.npy")
        xi = np.load(mc_data_path + "/xi.npy")
        xi_err = np.load(mc_data_path + "/xi_err.npy")

    return Ts, e, e_rr, mAbs, mAbs_err, magSusc, magSusc_err, binderCu, binderCu_err, k3, k3_err, xi, xi_err

#--------------------------------------------------------------------

def getGAN_Observables_DCGAN(Ts, res, samples, gen_new=False, injection=True):
    
    #------------------------
    if gen_new:    
        N = res * res

        gan_name = "Spin_DC_GAN"
        model_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "model-data")

        conditional_gan = importConditionalGAN(gan_name)

        latent_dim = 4096
        conditional_dim = 4

        epoch = 26
    
        #------------------------
        gan_model = load_ck(epoch, res, latent_dim, conditional_dim, conditional_gan, dc_ck_path, injection)
    
        #------------------------

        print("Generate getMC_Observables")
        if 1:
            m = list()
            e_raw = list()

            e = list()
            e_rr = list()
            mAbs = list()
            mAbs_err = list()
            magSusc = list()
            magSusc_err = list()
            binderCu = list()
            binderCu_err = list()
            k3 = list()
            k3_err = list()
            xi = list()
            xi_err = list()

        for T in Ts:
            print("T:", T)

            states = getStates_DCGAN(T, conditional_gan, gan_model, samples, conditional_dim, latent_dim)

            states = np.reshape(states, (-1, N))

            #calc observables
            g_energy = me.calc_states_energy(N, states)
            g_m     = np.sum(states, axis=-1) / N
            g_mAbs  = np.abs(g_m)
            g_m2    = np.square(g_m)
            g_mAbs3 = g_mAbs * g_m2
            g_m4    = np.square(g_m2)
        
            g_data_energy, g_data_mAbs, g_data_magSusc, g_data_binderCu, g_data_k3 = me.perform_observable_calculation(g_energy, g_m, g_mAbs, g_m2, g_mAbs3, g_m4, N, T)
            #g_data_energy, g_data_mAbs, g_data_magSusc, g_data_binderCu, g_data_k3 = me.perform_observable_calculation_non_binning(g_energy, g_m, g_mAbs, g_m2, g_mAbs3, g_m4, N, T)

            g_data_xi, g_data_xi_err = da.calc_spin_spin_correlation(states, N)

            #append
            if 1:
                m.append(g_m)
                e_raw.append(g_energy)

                e.append(g_data_energy.val)
                e_rr.append(g_data_energy.std)               
                mAbs.append(g_data_mAbs.val)
                mAbs_err.append(g_data_mAbs.std)
                magSusc.append(g_data_magSusc.val)
                magSusc_err.append(g_data_magSusc.std)
                binderCu.append(g_data_binderCu.val)
                binderCu_err.append(g_data_binderCu.std)
                k3.append(g_data_k3.val)
                k3_err.append(g_data_k3.std)
                xi.append(g_data_xi)
                xi_err.append(g_data_xi_err)

        np.save(dc_data_path + "/m.npy", m)
        np.save(dc_data_path + "/e_raw.npy", e_raw)

        np.save(dc_data_path + "/Ts.npy", Ts)
        np.save(dc_data_path + "/e.npy", e)
        np.save(dc_data_path + "/e_rr.npy", e_rr)      
        np.save(dc_data_path + "/mAbs.npy", mAbs)
        np.save(dc_data_path + "/mAbs_err.npy", mAbs_err)
        np.save(dc_data_path + "/magSusc.npy", magSusc)
        np.save(dc_data_path + "/magSusc_err.npy", magSusc_err)
        np.save(dc_data_path + "/binderCu.npy", binderCu)
        np.save(dc_data_path + "/binderCu_err.npy", binderCu_err)
        np.save(dc_data_path + "/k3.npy", k3)
        np.save(dc_data_path + "/k3_err.npy", k3_err)
        np.save(dc_data_path + "/xi.npy", xi)
        np.save(dc_data_path + "/xi_err.npy", xi_err)

    #------------------------
    else:
        print("Load getMC_Observables")
        m = np.load(dc_data_path + "/m.npy")
        e_raw = np.load(dc_data_path + "/e_raw.npy")

        Ts = np.load(dc_data_path + "/Ts.npy") 
        e = np.load(dc_data_path + "/e.npy")
        e_rr = np.load(dc_data_path + "/e_rr.npy")        
        mAbs = np.load(dc_data_path + "/mAbs.npy")
        mAbs_err = np.load(dc_data_path + "/mAbs_err.npy")
        magSusc = np.load(dc_data_path + "/magSusc.npy")
        magSusc_err = np.load(dc_data_path + "/magSusc_err.npy")
        binderCu = np.load(dc_data_path + "/binderCu.npy")
        binderCu_err = np.load(dc_data_path + "/binderCu_err.npy")
        k3 = np.load(dc_data_path + "/k3.npy")
        k3_err = np.load(dc_data_path + "/k3_err.npy")
        xi = np.load(dc_data_path + "/xi.npy")
        xi_err = np.load(dc_data_path + "/xi_err.npy")

    return Ts, e, e_rr, e_raw, m, mAbs, mAbs_err, magSusc, magSusc_err, binderCu, binderCu_err, k3, k3_err, xi, xi_err

#--------------------------------------------------------------------

def plot_m_e(data, g_data, Tc):
    Ts, e, e_rr, mAbs, mAbs_err, magSusc, magSusc_err, binderCu, binderCu_err, k3, k3_err, xi, xi_err = data
    g_Ts, g_e, g_e_rr, e_raw, g_m, g_mAbs, g_mAbs_err, g_magSusc, g_magSusc_err, g_binderCu, g_binderCu_err, g_k3, g_k3_err, g_xi, g_xi_err = g_data

    #--------------------------------------------------------------
    size=(13, 4.8)
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(1, 2, figure=fig)
    axs = np.array([fig.add_subplot(gs[0]), fig.add_subplot(gs[1])])

    
    title = [r"$\langle |m|\rangle$",  r"$\langle E\rangle$",
             r"$\chi$",                r"$\kappa_3$",
             r"$U_2$",                r"$\xi$"]

    mean = [mAbs, e]
    std  = [mAbs_err, e_rr]

    g_mean = [g_mAbs, g_e]
    g_std  = [g_mAbs_err, g_e_rr]
    
    clr_sim = "tab:blue"
    clr_gan = "tab:orange"

    for i in range(2):
        plt.sca(axs[i])
        plt.margins(0.03) 
        plt.axvline(Tc, color="gray", linestyle="--")
        plt.ylabel(title[i])
        plt.xlabel(r'$T/J$')

        #plt.errorbar(  Ts,   mean[i], fmt='.',   yerr=std[i], label="SIM", elinewidth=1, capsize=2, markersize=5, color=clr_sim)
        #plt.errorbar(g_Ts, g_mean[i], fmt='.', yerr=g_std[i], label="GAN", elinewidth=1, capsize=2, markersize=5, color=clr_gan)

        plt.fill_between(  Ts, np.array(mean[i])+np.array(std[i]), np.array(mean[i])-np.array(std[i]), alpha=0.2, color=clr_sim, lw=0)
        plt.fill_between(g_Ts, np.array(g_mean[i])+np.array(g_std[i]), np.array(g_mean[i])-np.array(g_std[i]), alpha=0.2, color=clr_gan, lw=0)

        plt.plot(  Ts,   mean[i], "-", color=clr_sim, lw=2.5, label="MC")
        plt.plot(g_Ts, g_mean[i], "--", color=clr_gan, lw=2.5, label="SpinGAN")
        
        #plt.plot(  Ts,   mean[i], "o", color=clr_sim)
        #plt.plot(g_Ts, g_mean[i], "o", color=clr_gan)
        if i ==1:
            plt.legend()

    #-------------------------------------------
    savePdf("gan_perf_m_e")
    savePng("gan_perf_m_e")
    saveSvg("gan_perf_m_e")
    return

def plot_chi_xi(data, g_data, Tc):
    Ts, e, e_rr, mAbs, mAbs_err, magSusc, magSusc_err, binderCu, binderCu_err, k3, k3_err, xi, xi_err = data
    g_Ts, g_e, g_e_rr, e_raw, g_m, g_mAbs, g_mAbs_err, g_magSusc, g_magSusc_err, g_binderCu, g_binderCu_err, g_k3, g_k3_err, g_xi, g_xi_err = g_data

    #--------------------------------------------------------------
    size=(13, 4.8)
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(1, 2, figure=fig)
    axs = np.array([fig.add_subplot(gs[0]), fig.add_subplot(gs[1])])

    title = [r"$\chi$", r"$\xi$"]

    mean = [magSusc, xi]
    std  = [magSusc_err, xi_err]

    g_mean = [g_magSusc, g_xi]
    g_std  = [g_magSusc_err, g_xi_err]
    
    clr_sim = "tab:blue"
    clr_gan = "tab:orange"

    for i in range(2):
        plt.sca(axs[i])
        plt.margins(0.03) 
        plt.axvline(Tc, color="gray", linestyle="--")
        plt.ylabel(title[i])
        plt.xlabel(r'$T/J$')

        #plt.errorbar(  Ts,   mean[i], fmt='.',   yerr=std[i], label="SIM", elinewidth=1, capsize=2, markersize=5, color=clr_sim)
        #plt.errorbar(g_Ts, g_mean[i], fmt='.', yerr=g_std[i], label="GAN", elinewidth=1, capsize=2, markersize=5, color=clr_gan)

        plt.fill_between(  Ts, np.array(mean[i])+np.array(std[i]), np.array(mean[i])-np.array(std[i]), alpha=0.2, color=clr_sim, lw=0)
        plt.fill_between(g_Ts, np.array(g_mean[i])+np.array(g_std[i]), np.array(g_mean[i])-np.array(g_std[i]), alpha=0.2, color=clr_gan, lw=0)

        plt.plot(  Ts,   mean[i], "-", color=clr_sim, lw=2.5, label="MC")
        plt.plot(g_Ts, g_mean[i], "--", color=clr_gan, lw=2.5, label="SpinGAN")

        #plt.plot(  Ts,   mean[i], "o", color=clr_sim)
        #plt.plot(g_Ts, g_mean[i], "o", color=clr_gan)
        if i ==1:
            plt.legend()

    #-------------------------------------------
    savePdf("gan_perf_chi_xi")
    savePng("gan_perf_chi_xi")
    saveSvg("gan_perf_chi_xi")
    return

#--------------------------------------------------------------------

def histo_plot_m(Ts, res):
    addpath = ""

    size=(13, 6)
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(2, 2, figure=fig)
    axs = np.array([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])])

    rows = cols = 2

    ticks = np.linspace(me.range_m[0], me.range_m[1], 5)
    empty_labels = ["" for x in ticks]

    clr_sim = "tab:blue"
    clr_gan = "tab:orange"

    x_Ts, e, e_rr, e_raw, m, mAbs, mAbs_err, magSusc, magSusc_err, binderCu, binderCu_err, k3, k3_err, xi, xi_err = getGAN_Observables_DCGAN(None, res, 0, gen_new=False)

    #-------------------------------------------
    for iy in range(rows):
        for ix in range(cols):
            i = iy * cols + ix

            plt.sca(axs[i])
            plt.margins(0.03)
           
            if ix == 0:
                plt.ylabel(r"Probability density")
            if iy == rows-1:
                plt.xlabel(r"$m$")
                plt.xticks(ticks)
            else:
                plt.xticks(ticks, empty_labels)
      
            T = Ts[i]
            te = r"$T/J$ = {t}".format(t=T)
            args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color="black")#, bbox=box)
            plt.text(0.08, 0.98 , te, args)

            #-------------------------
            t_energy, t_m, t_mAbs, t_m2, t_mAbs3, t_m4 = dh.load_spin_observables(T, addpath)

            data   = t_m           
            bins   = me.bin_size_m
            b_range = me.range_m

            plt.hist(data, bins=bins, range=b_range, density=True, label="MC", alpha=0.5, color=clr_sim)

            #-------------------------
            data = m[np.where(x_Ts == T)][0]

            plt.hist(data, bins=bins, range=b_range, density=True, label="SpinGAN", alpha=0.5, color=clr_gan)
    
            if i ==1:
                plt.legend()
    return

def histo_plot_e(Ts, res):
    addpath = ""

    size=(13, 6)
    fig = plt.figure(figsize=size, constrained_layout=True) 
    gs = plt.GridSpec(2, 2, figure=fig)
    axs = np.array([fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])])

    rows = cols = 2

    ticks = np.linspace(me.range_m[0], me.range_m[1], 5)
    empty_labels = ["" for x in ticks]

    clr_sim = "tab:blue"
    clr_gan = "tab:orange"

    x_Ts, e, e_rr, e_raw, m, mAbs, mAbs_err, magSusc, magSusc_err, binderCu, binderCu_err, k3, k3_err, xi, xi_err = getGAN_Observables_DCGAN(None, res, 0, gen_new=False)

    #-------------------------------------------
    for iy in range(rows):
        for ix in range(cols):
            i = iy * cols + ix

            plt.sca(axs[i])
            plt.margins(0.03)
           
            if ix == 0:
                plt.ylabel(r"Probability density")
            if iy == rows-1:
                plt.xlabel(r"$E$")
                plt.xticks(ticks)
            else:
                plt.xticks(ticks, empty_labels)
      
            T = Ts[i]
            te = r"$T/J$ = {t}".format(t=T)
            args = dict(horizontalalignment='left',verticalalignment='top', transform=plt.gca().transAxes, color="black")#, bbox=box)
            plt.text(0.08, 0.98 , te, args)

            #-------------------------
            t_energy, t_m, t_mAbs, t_m2, t_mAbs3, t_m4 = dh.load_spin_observables(T, addpath)

            data   = t_energy           
            bins   = me.bin_size_eng
            b_range = me.range_eng

            plt.hist(data, bins=bins, range=b_range, density=True, label="MS", alpha=0.5, color=clr_sim)

            #------------------------- 
            data = e_raw[np.where(x_Ts == T)][0]

            plt.hist(data, bins=bins, range=b_range, density=True, label="SpinGAN", alpha=0.5, color=clr_gan)
    
            if i ==1:
                plt.legend()

    return

#--------------------------------------------------------------------

def main():
    res = 64
    N   = res * res
    Tc = 1.0 * 2.0 / np.log(1.0 + np.sqrt(2.0))

    #MC temps
    Ts   = np.array([1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.3, 2.35, 2.4, 2.5, 2.6, 2.8, 3.0, 3.4])
    
    #interpolation temps
    g_Ts = np.linspace(1.0, 3.4, 100)
    g_Ts = np.concatenate([g_Ts,  Ts])
    g_Ts = np.sort(np.unique(g_Ts))

    #sampling of GAN
    samples = 2**14

    #-------------------------------------------
    data = getMC_Observables(Ts, N, gen_new=0)

    #-------------------------------------------
    g_data = getGAN_Observables_DCGAN(g_Ts, res, samples, gen_new=0)
       
    #-------------------------------------------
    plot_m_e(data, g_data, Tc)
    plot_chi_xi(data, g_data, Tc)

    #-------------------------------------------
    histo_plot_m([2.2, 2.3, 2.4, 2.8], res)
    save3("gan_hist_m")

    histo_plot_e([2.2, 2.3, 2.4, 2.8], res)
    save3("gan_hist_e")

    return

#--------------------------------------------------------------------

if __name__ == '__main__':
    main()

    plt.show()
#--------------------------------------------------------------------


