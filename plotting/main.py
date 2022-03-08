import numpy as np
import data_visualization
import model_evaluation
import matplotlib.pyplot as plt

#--------------------------------------------------------------------

#TODO: add single eval here, plot epoch history and mark the best one regarding the metric

def main() -> int:
  #-----------------------------------------------------------------
    
    latent_dims = {"Spin_DC_GAN" : 256}
    image_size = (64, 64, 1)

    model_names = np.array(["Spin_DC_GAN"])
    TJs         = np.array([1.0, 1.8, 2.0, 2.2, 2.25, 2.3, 2.4, 2.6, 3.4])
    
    epoch_min  = 20
    epoch_max  = 100
    epoch_step = 10

    #data cnt taken from GAN for evaluation
    images_count = 1000

    #-----------------------------------------------------------------
    #set this for evaluation of a single training run of a model in their folder (not the data folder) -> used for development
    if 1:
        single_eval = True     
        TJs         = np.array([2.6])

        epoch_step  = 3
        epoch_min   = 2*epoch_step
        epoch_max   = 99
        
    else:
        single_eval = False     

    #-----------------------------------------------------------------
    N = image_size[0] * image_size[1]
    epoch_cnt  = int((epoch_max-epoch_min) / epoch_step + 1)
    epochs     = np.linspace(epoch_min, epoch_max, epoch_cnt)

    for model_name in model_names:

        #--------------------------------------------------------------------
        #evaluate metrics for each epoch and determine/get the best data
        med_objs = model_evaluation.evaluate_model_metrics(TJs, model_name, epochs, latent_dims[model_name], image_size, images_count, N, singe_eval=single_eval)

        data_visualization.plot_performance_evaluation_hist(med_objs, use_energy_not_m=False)
        data_visualization.plot_performance_evaluation_hist(med_objs, use_energy_not_m=True)
        data_visualization.plot_performance_evaluation_phase(med_objs)

        #--------------------------------------------------------------------
        #next step is observalbe calc and plot
        mpd = model_evaluation.perform_data_processing(med_objs)

        data_visualization.plot_performance_evaluation_observables(TJs, mpd) 
        
        plt.show()

    return 0

if __name__ == '__main__':
    main()