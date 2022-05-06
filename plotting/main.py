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

    image_size = (64, 64, 1)

    model_names = np.array(["Spin_DC_GAN"])
    #TJs         = np.array([1.0, 1.8, 2.0, 2.2, 2.25, 2.3, 2.4, 2.6, 3.4])
    TJs         = np.array([1.0, 1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.3, 2.35, 2.4, 2.5, 2.6, 2.8, 3.0, 3.4])

    epoch_step = 1
    epoch_min  = 0 #epoch_step * 2
    epoch_max  = epoch_step * 1000
  
    #data cnt taken from GAN for evaluation
    images_count = 1000 #00

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
            med_objs, med_objs_interpolate = model_evaluation.evaluate_conditional_model_metrics(TJs, model_name, epochs, latent_dims[model_name], conditional_dims[model_name], image_size, images_count, N, single_eval=single_eval)
        
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

        #--------------------------------------------------------------------
        plt.show()

    return 0

if __name__ == '__main__':
    main()