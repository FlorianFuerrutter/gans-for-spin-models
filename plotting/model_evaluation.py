import numpy as np
import data_helper as dh
import data_analysis as da
from numba import jit, njit, prange

#--------------------------------------------------------------------

bin_size_mag = 40 #in [ 0, 1]
range_mag    = (0, 1)

bin_size_eng = 80 #in [-2, 0]
range_eng    = (-2, 0)

def create_hist(observable_name, data1, data2):
    bin_size = 10
    r        = (data1.min(), data1.max())

    if observable_name == "mag":
        bin_size = bin_size_mag
        r        = range_mag
    elif observable_name == "eng":
        bin_size = bin_size_eng
        r        = range_eng

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

def evaluate_model_metrics(TJs, model_name, epochs, latent_dim, image_size, images_count=1000, N=64*64):

    model_evaluation_data_list = []

    for TJ in TJs:
        #------------------------
        #get spin data
        energy, m, mAbs, m2, m4 = dh.load_spin_observables(TJ)

        #------------------------
        #get GAN data for all epochs to determine best epoch
        states_epoch = dh.generate_gan_data(TJ, model_name, epochs, images_count=images_count, latent_dim=latent_dim, image_size=image_size)

        g_energy = calc_states_epoch_energy(N, states_epoch)
        g_m    = np.sum(states_epoch, axis=2) / N
        g_mAbs = np.abs(g_m)
        g_m2   = np.square(g_m)
        g_m4   = np.square(g_m2)

        #------------------------
        #evaluate metrics of energy and abs(m)

        mag_pol = evaluate_metric_POL("mag", mAbs, g_mAbs)
        mag_emd = evaluate_metric_EMD("mag", mAbs, g_mAbs)

        eng_pol = evaluate_metric_POL("eng", energy, g_energy)
        eng_emd = evaluate_metric_EMD("eng", energy, g_energy)

        #------------------------
        #determine best epoch !! -> check how to combine emd and pol
        best_epoch_index = np.argmax(mag_pol)

        best_epoch = epochs[best_epoch_index]
        print("[evaluate_model_metrics] Model:", model_name, "TJ:", TJ, "Best epoch:", best_epoch)

        #------------------------
        #now extract data for this best_epoch
        gan_states = states_epoch[best_epoch_index]

        g_energy = g_energy[best_epoch_index]
        g_m      = g_m[best_epoch_index]
        g_mAbs   = g_mAbs[best_epoch_index]
        g_m2     = g_m2[best_epoch_index]
        g_m4     = g_m4[best_epoch_index]

        mag_pol = mag_pol[best_epoch_index]
        mag_emd = mag_emd[best_epoch_index]
        eng_pol = eng_pol[best_epoch_index]
        eng_emd = eng_emd[best_epoch_index]

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
        d.m4     = m4

        d.g_energy = g_energy
        d.g_m      = g_m
        d.g_mAbs   = g_mAbs
        d.g_m2     = g_m2
        d.g_m4     = g_m4
        
        d.best_epoch = best_epoch
        d.mag_pol = mag_pol
        d.mag_emd = mag_emd
        d.eng_pol = eng_pol
        d.eng_emd = eng_emd

        model_evaluation_data_list.append(d)

    return model_evaluation_data_list    

#--------------------------------------------------------------------

def perform_data_processing(med_objs : list[dh.model_evaluation_data]):
    mpd = dh.model_processed_data()
    mpd.model_name = med_objs[-1].model_name

    #has same len as TJs
    for d in med_objs:
        
        #----------------------------------------
        #compute values for MC values -> binning

        #-------------------
        mean, err, corr = da.binningAnalysisSingle(d.energy)
        mpd.energy.append(dh.err_data(mean, err))

        #mean, err, corr = da.binningAnalysisSingle(d.m)
        #mpd.m.append(dh.err_data(mean, err))
     
        mean, err, corr = da.binningAnalysisSingle(d.mAbs)
        mpd.mAbs.append(dh.err_data(mean, err))

        #-------------------
        meanMAbs, errorMAbs, meanM2, errorM2, meanM4, errorM4, corr, binMAbs, binM2, binM4 = da.binningAnalysisTriple(d.mAbs, d.m2, d.m4)

        meanMagSusc, errorMagSusc   = da.mSuscJackknife(binMAbs, binM2, d.N, d.T)
        mpd.magSusc.append(dh.err_data(meanMagSusc, errorMagSusc))
        
        meanBinderCu, errorBinderCu = da.mBinderCuJackknife(binM2, binM4)
        mpd.binderCu.append(dh.err_data(meanBinderCu, errorBinderCu))

        #----------------------------------------
        #compute values for GAN values -> binning

        #-------------------
        mean, err, corr = da.binningAnalysisSingle(d.g_energy)
        mpd.g_energy.append(dh.err_data(mean, err))
        
        #mean, err, corr = da.binningAnalysisSingle(d.g_m)
        #mpd.g_m.append(dh.err_data(mean, err))

        mean, err, corr = da.binningAnalysisSingle(d.g_mAbs)
        mpd.g_mAbs.append(dh.err_data(mean, err))

        #-------------------
        meanMAbs, errorMAbs, meanM2, errorM2, meanM4, errorM4, corr, binMAbs, binM2, binM4 = da.binningAnalysisTriple(d.g_mAbs, d.g_m2, d.g_m4)

        meanMagSusc, errorMagSusc   = da.mSuscJackknife(binMAbs, binM2, d.N, d.T)
        mpd.g_magSusc.append(dh.err_data(meanMagSusc, errorMagSusc))
        
        meanBinderCu, errorBinderCu = da.mBinderCuJackknife(binM2, binM4)
        mpd.g_binderCu.append(dh.err_data(meanBinderCu, errorBinderCu))

    return mpd