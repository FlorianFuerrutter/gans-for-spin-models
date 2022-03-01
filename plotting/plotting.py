import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------
#-------------------------------------

def reduceIntoBin2(data):
    binnedSize = data.shape[0] // 2
    binnedData = np.zeros(binnedSize)
    
    for j in range(binnedSize):
        binnedData[j] = data[2*j] + data[2*j + 1]

    return binnedData * 0.5

def binningAnalysisSingle(data, printAlreadyConvergedWarning=False):
    #algo find max of errors -> use this binsize
    #if not converging -> max is last element then report error converging

    N = data.shape[0]
    maxBinningSteps = int(np.log2(N)) #so last binning has 2 elements, 1 element is useless for error

    mean       = np.zeros(maxBinningSteps)
    meanErrors = np.zeros(maxBinningSteps)

    #starting data
    binnedData    = data
    mean[0]       = np.mean(data)
    meanErrors[0] = np.std(data) / np.sqrt(N)

    #binning  up to maxBinningSteps
    for i in range(1, maxBinningSteps):
        #binsize = 2**i

        #binning step
        binnedData = reduceIntoBin2(binnedData)
        N = binnedData.shape[0] 

        #new error, mean
        mean[i]       = np.mean(binnedData)
        meanErrors[i] = np.std(binnedData) / np.sqrt(N)

    maxElement = np.argmax(meanErrors)
    if (maxElement+1) == maxBinningSteps: 
        info = "   (elements in bin of max error: " + str(data.shape[0] // 2**maxElement) + ")"
        print("binningAnalysisSingle: [NOT CONVERGED]  increase dataset," + info)
    if maxElement == 0 and printAlreadyConvergedWarning:
        info = "   (elements in bin of max error: " + str(data.shape[0] // 2**maxElement) + ")"
        print("binningAnalysisSingle: [Already CONVERGED]  first error is largest," + info)

    #print("max error at binstep=%d, binsize=%d" % (mi, 2**mi))  
    corrSize = 2**maxElement
    return mean[maxElement], meanErrors[maxElement], corrSize

#-------------------------------------
#-------------------------------------

import sys, os.path
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + "/models/Spin_StyleGAN2"
sys.path.append(model_path)
import gan

def generate_gan_data(TJ):
    

    #load gan model
    #load weights for tj
    #generate spin data

    #clip to +-1


def do():
    #do for each TJ                 -> generate data
        generate_gan_data()

       
    #do for each TJ                 -> convert data into value
        #load MC spin data
        #compute values for MC values -> binning
        #plot these

        #load GAN spin data
        #compute values for GAN values -> binning
        #plot these



#-------------------------------------
#-------------------------------------

def main() -> int:   
    return 0

if __name__ == '__main__':
    main()