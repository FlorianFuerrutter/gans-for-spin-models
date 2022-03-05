import numpy as np
import data_helper as dh
import data_analysis as da

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

def evaluate_model_metrics(TJs, model_name, epochs, latent_dim, image_size, images_count=1000, N=64*64):

    model_evaluation_data_list = []

    for TJ in TJs:
        #------------------------
        #get spin data
        energy, m2 = dh.load_spin_observables(TJ) #TODO OBSERVABLES !!!!!!!!
        mAbs = np.sqrt(m2)

        #------------------------
        #get GAN data for all epochs to determine best epoch
        states_epoch = dh.generate_gan_data(TJ, model_name, epochs, images_count=images_count, latent_dim=latent_dim, image_size=image_size)

        g_mAbs = np.abs(np.sum(states_epoch, axis=2)) / N
        #g_energy = []

        #------------------------
        #evaluate metrics of energy and abs(m)

        mag_pol = evaluate_metric_POL("mag", mAbs, g_mAbs)
        mag_emd = evaluate_metric_EMD("mag", mAbs, g_mAbs)

        eng_pol = None #evaluate_metric_POL("eng", energy, g_energy)
        eng_emd = None #evaluate_metric_EMD("eng", energy, g_energy)

        #------------------------
        #determine best epoch !! -> check how to combine emd and pol
        best_epoch_index = np.argmax(mag_pol)

        best_epoch = epochs[best_epoch_index]
        print("best_epoch", best_epoch)

        #------------------------
        #now extract data for this best_epoch
        gan_states = states_epoch[best_epoch_index]

        g_mAbs   = g_mAbs[best_epoch_index]
        g_energy = None #g_energy[best_epoch_index]

        mag_pol = mag_pol[best_epoch_index]
        mag_emd = mag_emd[best_epoch_index]

        #eng_pol = eng_pol[best_epoch_index]
        #eng_pol = eng_pol[best_epoch_index]

        #------------------------
        #set return obj
        d = dh.model_evaluation_data(mAbs, energy, gan_states, g_mAbs, g_energy, best_epoch, mag_pol, mag_emd, eng_pol, eng_emd)
        model_evaluation_data_list.append(d)

    return model_evaluation_data_list    

#--------------------------------------------------------------------

def perform_data_processing(med_objs : list[dh.model_evaluation_data]):
    mpd = dh.model_processed_data()

    #has same len as TJs
    for d in med_objs:
        
        #----------------------------------------
        #compute values for MC values -> binning

        mean, err, corr = da.binningAnalysisSingle(d.mAbs)
        mpd.mAbs.append(dh.err_data(mean, err))

        mean, err, corr = da.binningAnalysisSingle(d.energy)
        mpd.energy.append(dh.err_data(mean, err))

        #XXXXX
        #TODO do triple for sucsebility and binder ratio
        #XXXXX

        #----------------------------------------
        #compute values for GAN values -> binning

        mean, err, corr = da.binningAnalysisSingle(d.g_mAbs)
        mpd.g_mAbs.append(dh.err_data(mean, err))

        #mean, err, corr = da.binningAnalysisSingle(d.g_energy)
        #mpd.g_energy.append(dh.err_data(mean, err))

    return mpd