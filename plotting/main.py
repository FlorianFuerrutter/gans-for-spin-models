import numpy as np
import data_visualization
import model_evaluation
import matplotlib.pyplot as plt

#--------------------------------------------------------------------

def main() -> int:
  #-----------------------------------------------------------------
    
    latent_dims      = {"Spin_DC_GAN" : 4096}
    conditional      = {"Spin_DC_GAN" : 1}
    conditional_dims = {"Spin_DC_GAN" : 4}
    model_names = np.array(["Spin_DC_GAN"])

    #latent_dims      = {"Spin_StyleGAN2" : 4096}
    #conditional      = {"Spin_StyleGAN2" : 1}
    #conditional_dims = {"Spin_StyleGAN2" : 4}
    #model_names = np.array(["Spin_StyleGAN2"])

    image_size = (64, 64, 1)
    addpath = "" #"L48/"

    TJs         = np.array([1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.3, 2.35, 2.4, 2.5, 2.6, 2.8, 3.0, 3.4])

    epoch_step = 1
    epoch_min  = 0
    epoch_max  = 1000
  
    #data cnt taken from GAN for evaluation
    images_count = 1024 #00

    #-----------------------------------------------------------------
    #set this for evaluation of a single training run of a model in their folder (not the data folder) -> used for development
    if 0:
        single_eval = True  #sets load path for weights
        TJs         = np.array([1.8])

        epoch_step  = 3
        epoch_min   = epoch_step * 2
        epoch_max   = epoch_step * 100 
        
    else:
        single_eval = False     

    #-----------------------------------------------------------------
    N = image_size[0] * image_size[1]
    epoch_cnt  = int((epoch_max-epoch_min) / epoch_step + 1)
    epochs     = np.linspace(epoch_min, epoch_max, epoch_cnt)

    for model_name in model_names:

        #--------------------------------------------------------------------
        #evaluate metrics for each epoch and determine/get the best data
        mpd_interpolate = None

        if conditional[model_name]:
            med_objs, med_objs_interpolate = model_evaluation.evaluate_conditional_model_metrics(TJs, model_name, epochs, latent_dims[model_name], conditional_dims[model_name], image_size, images_count, N, single_eval=single_eval, addpath=addpath)
        
            mpd_interpolate = model_evaluation.perform_data_processing(med_objs_interpolate, False)
        else:
            med_objs = model_evaluation.evaluate_model_metrics(TJs, model_name, epochs, latent_dims[model_name], image_size, images_count, N, single_eval=single_eval)

        #--------------------------------------------------------------------
        #plot hists
        data_visualization.plot_performance_evaluation_hist(med_objs, use_energy_not_m=False)
        data_visualization.plot_performance_evaluation_hist(med_objs, use_energy_not_m=True)
        data_visualization.plot_performance_evaluation_phase(med_objs)

        #--------------------------------------------------------------------

        mpd = model_evaluation.perform_data_processing(med_objs)

        data_visualization.plot_performance_evaluation_observables(TJs, mpd, mpd_interpolate) 

        #store the data for later
        if 1:
            save_path = "obs/"
            names = ["mAbs", "energy", "magSusc", "k3", "binderCu", "xi"]

            mc_data_list  = [mpd.mAbs  , mpd.energy,   mpd.magSusc,   mpd.k3, mpd.binderCu, mpd.xi]
            gan_data_list = [mpd.g_mAbs, mpd.g_energy, mpd.g_magSusc, mpd.g_k3, mpd.g_binderCu, mpd.g_xi] 


            for i in range(6):
                mc_data  = mc_data_list[i]
                gan_data = gan_data_list[i]
                name = names[i]

                mean, err, std       = [x.val for x in mc_data], [x.err for x in mc_data], [x.std for x in mc_data]
                g_mean, g_err, g_std = [x.val for x in gan_data], [x.err for x in gan_data], [x.std for x in gan_data]

                np.save(save_path + name + "_mc_mean", mean)
                np.save(save_path + name + "_mc_std", std)

                np.save(save_path + name + "_gan_mean", g_mean)
                np.save(save_path + name + "_gan_std", g_std)

        #--------------------------------------------------------------------
        plt.show()

    return 0

if __name__ == '__main__':
    main()