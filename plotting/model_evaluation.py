import numpy as np
import data_helper as dh
import data_analysis as da
from numba import jit, njit, prange

#--------------------------------------------------------------------

bins_per_unit = 40

bin_size_m = bins_per_unit*2 #in [ 0, 1]
range_m    = (-1, 1)

bin_size_mAbs = bins_per_unit*1 #in [ 0, 1]
range_mAbs    = (0, 1)

bin_size_eng = int(bins_per_unit*2.5) #in [-2, 2]
range_eng    = (-2, 0.5)

#-------- 1D ---------
def create_hist(observable_name, data1, data2):
    bin_size = 10
    r        = (data1.min(), data1.max())

    if observable_name == "mAbs":
        bin_size = bin_size_mAbs
        r        = range_mAbs
    elif observable_name == "eng":
        bin_size = bin_size_eng
        r        = range_eng
    elif observable_name == "m":
        bin_size = bin_size_m
        r        = range_m

    hist1, bin_edges1 = np.histogram(data1, bin_size, range=r, density=True)
    hist2, bin_edges2 = np.histogram(data2, bin_size, range=r, density=True)

    assert np.allclose(bin_edges1, bin_edges2)

    return hist1, hist2, bin_edges1

def evaluate_metric_POL(observable_name, spin_data, gan_data):

    pol = []
    for i in range(gan_data.shape[0]):
        hist1, hist2, bin_edges = create_hist(observable_name, spin_data, gan_data[i])
    
        #calc %OL
        p1 = hist1 * np.diff(bin_edges)
        p2 = hist2 * np.diff(bin_edges)

        pol.append(np.sum(np.minimum(p1, p2)))

    return np.array(pol)

def evaluate_metric_EMD(observable_name, spin_data, gan_data):

    emd_list = []
    for i in range(gan_data.shape[0]):
        hist1, hist2, bin_edges = create_hist(observable_name, spin_data, gan_data[i])

        #calc EMD
        p1 = hist1 * np.diff(bin_edges)
        p2 = hist2 * np.diff(bin_edges)
        diff = p1 - p2

        emd = 0.0
        for x in range(1, hist1.size + 1):
            emd += np.abs(np.sum(diff[:x]))

        emd_list.append(emd)

    return np.array(emd_list)
 
#-------- 2D ---------
def create_hist2D(spin_data_m, spin_data_energy, gan_data_m, gan_data_energy, bin_scale=1):

    #------------------------
    #spin hist
    x_spin = spin_data_m
    y_spin = spin_data_energy

    H_spin, xedges_spin, yedges_spin = np.histogram2d(x_spin, y_spin, bins=(int(bin_size_m*bin_scale), int(bin_size_eng*bin_scale)), range=[range_m, range_eng], density=True)
    H_spin = H_spin.T #to Cartesian 

    #------------------------
    #gan hist
    x_gan = gan_data_m
    y_gan = gan_data_energy

    H_gan, xedges_gan, yedges_gan = np.histogram2d(x_gan, y_gan, bins=(int(bin_size_m*bin_scale), int(bin_size_eng*bin_scale)), range=[range_m, range_eng], density=True)
    H_gan = H_gan.T #to Cartesian 

    #------------------------
    assert np.allclose(xedges_spin, xedges_gan)
    assert np.allclose(yedges_spin, yedges_gan)

    return H_spin, H_gan, xedges_spin, yedges_spin

def evaluate_metric_EM_phase_POL(spin_data_m, spin_data_energy, gan_data_m, gan_data_energy):
    assert gan_data_m.shape[0] == gan_data_energy.shape[0]
    
    pol = []
    for i in range(gan_data_m.shape[0]):

        H_spin, H_gan, xedges, yedges = create_hist2D(spin_data_m, spin_data_energy, gan_data_m[i], gan_data_energy[i])
   
        #calc %OL
        A = np.outer(np.diff(yedges), np.diff(xedges))

        p1 = H_spin * A
        p2 = H_gan  * A

        pol.append(np.sum(np.minimum(p1, p2)))

    return np.array(pol)

#--------------------------------------------------------------------

def evaluate_metric_OOL(energy, m, mAbs, m2, mAbs3, m4, g_energy, g_m, g_mAbs, g_m2, g_mAbs3, g_m4, N, T): #observable overlap
    ''' doing for m, E, susc, k3, U2 '''

    #-----------------------------
    #calc MC values
    data_energy, data_mAbs, data_magSusc, data_binderCu, data_k3 = perform_observable_calculation(energy, m, mAbs, m2, mAbs3, m4, N, T) 

    #-----------------------------
    #calc GAN epoch values

    gan_eng  = []
    gan_mAbs = []
    gan_susc = []
    gan_bind = []
    gan_k3   = []
    for i in range(g_energy.shape[0]):
        g_data_energy, g_data_mAbs, g_data_magSusc, g_data_binderCu, g_data_k3 = perform_observable_calculation(g_energy[i], g_m[i], g_mAbs[i], g_m2[i], g_mAbs3[i], g_m4[i], N, T) 

        gan_eng.append(g_data_energy)
        gan_mAbs.append(g_data_mAbs)
        gan_susc.append(g_data_magSusc)
        gan_bind.append(g_data_binderCu)
        gan_k3.append(g_data_k3)
 
    #-----------------------------
    #calc the metric -> here the gan mean distance in units of the error to the MC mean
    # metric is then the mean distance over the obs for one epoch

    eps = 1e-14 #for numerical reasons
    distances = []
    for i in range(g_energy.shape[0]):
        
        d1 = ( gan_eng[i].val  - data_energy.val   ) / ( gan_eng[i].err  + eps )
        d2 = ( gan_mAbs[i].val - data_mAbs.val     ) / ( gan_mAbs[i].err + eps )
        d3 = ( gan_susc[i].val - data_magSusc.val  ) / ( gan_susc[i].err + eps )
        d4 = ( gan_bind[i].val - data_binderCu.val ) / ( gan_bind[i].err + eps )
        d5 = ( gan_k3[i].val   - data_k3.val       ) / ( gan_k3[i].err   + eps )

        if 0:
            print("gan_eng[i].val", gan_eng[i].val)
            print("data_energy.val", data_energy.val)
            print("distance gan_eng", d1)
            print("\n")

        distances.append(np.square([d1, d2, d3, d4, d5]))

    metric = np.sqrt(np.mean(distances, axis=1))

    return metric

#--------------------------------------------------------------------

#hamilton used, later load B image for energy calculation!!!!
def calc_states_epoch_energy(N, states_epoch):
    J = 1
    L = int(np.sqrt(N))

    #------------------

    def genNNList(L):
        N = L**2
        nn1d = np.zeros((N, 4), dtype=int)
        nn2d = np.zeros((L, L, 4, 2), dtype=int) 
        for y in range(L):
            for x in range(L):
                nn2d[y, x][0] = np.array([y  , x+1]) % L  #right
                nn2d[y, x][1] = np.array([y  , x-1]) % L  #left
                nn2d[y, x][2] = np.array([y-1, x  ]) % L  #top
                nn2d[y, x][3] = np.array([y+1, x  ]) % L  #bot

                nn1d[y*L+x] = nn2d[y, x, :, 0]*L + nn2d[y, x, :, 1]
            
        return nn1d
    nn = genNNList(L)

    @njit(cache=True)
    def calc_state_energy(state, nn, J, N):
        energy = 0.0
        for i in prange(N):
            nnI   = nn[i]

            local  = 0.0
            local += state[nnI[0]];
            local += state[nnI[1]];
            local += state[nnI[2]];
            local += state[nnI[3]];
            energy += local * state[i];
        return - 0.5 * J * energy #0.5 bcs of double counting

    #------------------
    epoch_energies = []
    for states in states_epoch:

        states_energy = []
        for state in states:
            e = calc_state_energy(state, nn, J, N) / N
            states_energy.append(e)

        epoch_energies.append(states_energy)

    return np.array(epoch_energies)

def evaluate_model_metrics(TJs, model_name, epochs, latent_dim, image_size, images_count=1000, N=64*64, singe_eval=False):

    model_evaluation_data_list = []

    for TJ in TJs:
        #------------------------
        #get spin data
        energy, m, mAbs, m2, mAbs3, m4 = dh.load_spin_observables(TJ)

        #------------------------
        #get GAN data for all epochs to determine best epoch
        states_epoch = dh.generate_gan_data(TJ, model_name, epochs, images_count=images_count, latent_dim=latent_dim, image_size=image_size, alt_path=singe_eval)

        g_energy = calc_states_epoch_energy(N, states_epoch)
        g_m     = np.sum(states_epoch, axis=2) / N
        g_mAbs  = np.abs(g_m)
        g_m2    = np.square(g_m)
        g_mAbs3 = g_mAbs * g_m2
        g_m4    = np.square(g_m2)

        obs_dist = evaluate_metric_OOL(energy, m, mAbs, m2, mAbs3, m4, g_energy, g_m, g_mAbs, g_m2, g_mAbs3, g_m4, N, TJ)

        #------------------------
        #evaluate metrics of m, energy and abs(m)

        m_pol = evaluate_metric_POL("m", m, g_m)
        m_emd = evaluate_metric_EMD("m", m, g_m)

        mAbs_pol = evaluate_metric_POL("mAbs", mAbs, g_mAbs)
        mAbs_emd = evaluate_metric_EMD("mAbs", mAbs, g_mAbs)

        eng_pol = evaluate_metric_POL("eng", energy, g_energy)
        eng_emd = evaluate_metric_EMD("eng", energy, g_energy)

        phase_pol = evaluate_metric_EM_phase_POL(m, energy, g_m, g_energy)
       
        #------------------------
        #determine best epoch !! -> check how to combine emd and pol
        #combine m_pol+eng_pol or direclty use phase_pol??
        deval = (mAbs_pol + 2.0*phase_pol) / 3.0

        print("\n")
        print("[evaluate_model_metrics] m_pol at epoch:", epochs[np.argmax(m_pol)])
        print("[evaluate_model_metrics] mAbs_pol at epoch:", epochs[np.argmax(mAbs_pol)])
        print("[evaluate_model_metrics] eng_pol at epoch:", epochs[np.argmax(eng_pol)])
        print("[evaluate_model_metrics] phase_pol at epoch:", epochs[np.argmax(phase_pol)])
        print("[evaluate_model_metrics] deval at epoch:", epochs[np.argmax(deval)])
        print("\n")
        print("[evaluate_model_metrics] m_emd at epoch:", epochs[np.argmin(m_emd)])
        print("[evaluate_model_metrics] mAbs_emd at epoch:", epochs[np.argmin(mAbs_emd)])
        print("[evaluate_model_metrics] eng_emd at epoch:", epochs[np.argmin(eng_emd)])
        print("\n")
        print("[evaluate_model_metrics] obs_dist at epoch:", epochs[np.argmin(obs_dist)], " = ", obs_dist[np.argmin(obs_dist)])
        print("\n")

        best_epoch_index1 = np.argmax(deval)
        best_epoch_index2 = np.argmax(phase_pol)
        best_epoch_index3 = np.argmax(eng_pol)
        best_epoch_index4 = np.argmax(mAbs_pol)
        best_epoch_index5 = np.argmin(eng_emd)

        best_epoch_index6 = np.argmin(obs_dist)

        #check how to determine the BEST!!!
        best_epoch_index = best_epoch_index6
      
        #------------------------
        best_epoch = epochs[best_epoch_index]
        print("[evaluate_model_metrics] Model:", model_name, "TJ:", TJ, "Best epoch:", best_epoch, "with percent OL (m_pol):", m_pol[best_epoch_index])
        print("[evaluate_model_metrics] deval at epoch:", epochs[best_epoch_index], "is:", deval[best_epoch_index])
        print("[evaluate_model_metrics] phase_POl at epoch:", epochs[best_epoch_index], "is:", phase_pol[best_epoch_index])

        #------------------------
        #now extract data for this best_epoch
        gan_states = states_epoch[best_epoch_index]

        if 0:
            import matplotlib.pyplot as plt
            import os

            #-----
            dark_image_grey1 = np.reshape(gan_states, (-1,64,64))
            dark_image_grey_fourier1 = np.fft.fftshift(np.fft.fft2(dark_image_grey1), axes=(1,2))        
            #plt.figure(figsize=(8, 6), dpi=80)
            #plt.imshow(np.log(abs(dark_image_grey_fourier1[0])), cmap='gray')

            #-----
            print("loadtxt")
            path = os.path.join(os.path.dirname(__file__), "..", "data", "train")
            file_path = os.path.join(path, "simulation_states_TJ_2.25.txt")
            states = np.loadtxt(file_path, skiprows=1, dtype=np.float32)[:1000]
            states = np.reshape(states, ( -1, 64, 64, 1))

            dark_image_grey2 = np.reshape(states, (-1,64,64))
            dark_image_grey_fourier2 = np.fft.fftshift(np.fft.fft2(dark_image_grey2), axes=(1,2))        
            #plt.figure(figsize=(8, 6), dpi=80)
            #plt.imshow(np.log(abs(dark_image_grey_fourier2[0])), cmap='gray')

            #----- ftt, mul, sum
            mult_time = abs(dark_image_grey_fourier1 * dark_image_grey_fourier2)
            #plt.figure(figsize=(8, 6), dpi=80)
            #plt.imshow(np.log(abs(mult_time[0])), cmap='gray')
            sim = np.sum(mult_time, axis=(1,2)) / 4096.0
            sim_max = np.amax(sim)
            print("freq mult: ", sim_max)

            #----- ftt, min, sum
            mult_time2 = np.minimum(abs(dark_image_grey_fourier1), abs(dark_image_grey_fourier2))
            #plt.figure(figsize=(8, 6), dpi=80)
            #plt.imshow(np.log(abs(mult_time2[0])), cmap='gray')
            sim = np.sum(mult_time2, axis=(1,2)) / 4096.0
            sim_max = np.amax(sim)
            print("freq min: ", sim_max)

            #----- mul, fft
            mult_space = dark_image_grey1 * dark_image_grey2
            dark_image_grey_fourier_mult_space = np.fft.fftshift(np.fft.fft2(mult_space), axes=(1,2))  
            #plt.figure(figsize=(8, 6), dpi=80)
            #plt.imshow(np.log(abs(dark_image_grey_fourier_mult_space[0])), cmap='gray')

            #plt.show()

        #------------------------

        g_energy = g_energy[best_epoch_index]
        g_m      = g_m[best_epoch_index]
        g_mAbs   = g_mAbs[best_epoch_index]
        g_m2     = g_m2[best_epoch_index]
        g_mAbs3  = g_mAbs3[best_epoch_index]
        g_m4     = g_m4[best_epoch_index]

        m_pol   = m_pol[best_epoch_index]
        m_emd   = m_emd[best_epoch_index]
        mAbs_pol = mAbs_pol[best_epoch_index]
        mAbs_emd = mAbs_emd[best_epoch_index]
        eng_pol = eng_pol[best_epoch_index]
        eng_emd = eng_emd[best_epoch_index]
        
        phase_pol = phase_pol[best_epoch_index]
        obs_dist  = obs_dist[best_epoch_index]

        #------------------------
        #set return obj
        d = dh.model_evaluation_data()
        d.T = TJ
        d.N = N
        d.model_name = model_name

        d.energy = energy
        d.m      = m
        d.mAbs   = mAbs
        d.m2     = m2
        d.mAbs3  = mAbs3
        d.m4     = m4

        d.g_energy = g_energy
        d.g_m      = g_m
        d.g_mAbs   = g_mAbs
        d.g_m2     = g_m2
        d.g_mAbs3  = g_mAbs3
        d.g_m4     = g_m4
        
        d.best_epoch = best_epoch
        d.m_pol   = m_pol
        d.m_emd   = m_emd
        d.mAbs_pol = mAbs_pol
        d.mAbs_emd = mAbs_emd
        d.eng_pol = eng_pol
        d.eng_emd = eng_emd
        
        d.phase_pol = phase_pol
        d.obs_dist  = obs_dist

        model_evaluation_data_list.append(d)

    return model_evaluation_data_list    

#--------------------------------------------------------------------

def perform_observable_calculation(energy, m, mAbs, m2, mAbs3, m4, N, T):  
    mean, err, corr = da.binningAnalysisSingle(energy)
    data_energy = dh.err_data(mean, err)

    #mean, err, corr = da.binningAnalysisSingle(m)
    #data_m = dh.err_data(mean, err)

    mean, err, corr = da.binningAnalysisSingle(mAbs)
    data_mAbs = dh.err_data(mean, err)

    #-------------------
    meanMAbs, errorMAbs, meanM2, errorM2, meanM4, errorM4, corr, binMAbs, binM2, binM4 = da.binningAnalysisTriple(mAbs, m2, m4)

    meanMagSusc, errorMagSusc = da.mSuscJackknife(binMAbs, binM2, N, T)
    data_magSusc = dh.err_data(meanMagSusc, errorMagSusc)

    meanBinderCu, errorBinderCu = da.mBinderCuJackknife(binM2, binM4)
    data_binderCu = dh.err_data(meanBinderCu, errorBinderCu)

    #-------------------
    meanMAbs, errorMAbs, meanM2, errorM2, meanMAbs3, errorMAbs3, corr, binMAbs, binM2, binMAbs3 = da.binningAnalysisTriple(mAbs, m2, mAbs3)

    meanK3, errorK3 = da.mK3Jackknife(binMAbs, binM2, binMAbs3, N, T)
    data_k3 = dh.err_data(meanK3, errorK3)

    return data_energy, data_mAbs, data_magSusc, data_binderCu, data_k3


def perform_data_processing(med_objs : list[dh.model_evaluation_data]):
    mpd = dh.model_processed_data()
    mpd.model_name    = med_objs[-1].model_name
    mpd.model_name_id = med_objs[-1].model_name_id
    mpd.obs_dist      = med_objs[-1].obs_dist

    #has same len as TJs
    for d in med_objs:
        #----------------------------------------
        #compute values for MC values -> binning

        data_energy, data_mAbs, data_magSusc, data_binderCu, data_k3 = perform_observable_calculation(d.energy, d.m, d.mAbs, d.m2, d.mAbs3, d.m4, d.N, d.T)

        mpd.energy.append(data_energy)
        #mpd._m.append(data_m)
        mpd.mAbs.append(data_mAbs)
        mpd.magSusc.append(data_magSusc)
        mpd.binderCu.append(data_binderCu)
        mpd.k3.append(data_k3)

        #----------------------------------------
        #compute values for GAN values -> binning

        g_data_energy, g_data_mAbs, g_data_magSusc, g_data_binderCu, g_data_k3 = perform_observable_calculation(d.g_energy, d.g_m, d.g_mAbs, d.g_m2, d.g_mAbs3, d.g_m4, d.N, d.T)

        mpd.g_energy.append(g_data_energy)
        #mpd.g_m.append(g_data_m)
        mpd.g_mAbs.append(g_data_mAbs)
        mpd.g_magSusc.append(g_data_magSusc)
        mpd.g_binderCu.append(g_data_binderCu)
        mpd.g_k3.append(g_data_k3)

    return mpd