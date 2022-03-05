import numpy as np
import data_visualization
import model_evaluation

#--------------------------------------------------------------------

def main() -> int:

    latent_dim = 128
    image_size = (64, 64, 1)

    model_names = np.array(["Spin_DC_GAN"])
    TJs         = np.array([1.0, 1.8, 2.0, 2.2, 2.4, 2.6, 3.4])
    TJs = np.array([1.8])

    epoch_min  = 20
    epoch_max  = 90
    epoch_step = 10

    #data cnt taken from GAN for evaluation
    images_count = 1000

    #-----------------------------------------------------------------
    N = image_size[0] * image_size[1]
    epoch_cnt  = int((epoch_max-epoch_min) / epoch_step + 1)
    epochs     = np.linspace(epoch_min, epoch_max, epoch_cnt)

    for model_name in model_names:

        #evaluate metrics for each epoch and determine/get the best data
        med_objs = model_evaluation.evaluate_model_metrics(TJs, model_name, epochs, latent_dim, image_size, images_count, N)

        #next step is observalbe calc and plot
        mpd = model_evaluation.perform_data_processing(med_objs)

        #plot the data
        data_visualization.plot_performance_evaluation(TJs, mpd)

    return 0

if __name__ == '__main__':
    main()