import numpy as np
from numba import jit, njit, prange

#--------------------------------------------------------------------

@njit(cache=True)
def reduceIntoBin2(data):
    binnedSize = data.shape[0] // 2
    binnedData = np.zeros(binnedSize)
    
    for j in prange(binnedSize):
        binnedData[j] = data[2*j] + data[2*j + 1]

    return binnedData * 0.5

def binningAnalysisSingle(data, printAlreadyConvergedWarning=False):
    #algo find max of errors -> use this binsize
    #if not converging -> max is last element then report error converging

    N = data.shape[0]
    maxBinningSteps = int(np.log2(N)) #so last binning has 2 elements, 1 element is useless for error

    mean       = np.zeros(maxBinningSteps)
    meanErrors = np.zeros(maxBinningSteps)
    stds       = np.zeros(maxBinningSteps)

    #starting data
    binnedData    = data
    mean[0]       = np.mean(data)
    meanErrors[0] = np.std(data) / np.sqrt(N)
    stds[0]       = np.std(data)

    #binning  up to maxBinningSteps
    for i in range(1, maxBinningSteps):
        #binsize = 2**i

        #binning step
        binnedData = reduceIntoBin2(binnedData)
        N = binnedData.shape[0] 

        #new error, mean
        mean[i]       = np.mean(binnedData)
        meanErrors[i] = np.std(binnedData) / np.sqrt(N)
        stds[i]       = np.std(binnedData)

    maxElement = np.argmax(meanErrors)
    if (maxElement+1) == maxBinningSteps: 
        info = "   (elements in bin of max error: " + str(data.shape[0] // 2**maxElement) + ")"
        print("binningAnalysisSingle: [NOT CONVERGED]  increase dataset," + info)
    if maxElement == 0 and printAlreadyConvergedWarning:
        info = "   (elements in bin of max error: " + str(data.shape[0] // 2**maxElement) + ")"
        print("binningAnalysisSingle: [Already CONVERGED]  first error is largest," + info)

    #print("max error at binstep=%d, binsize=%d" % (mi, 2**mi))  
    corrSize = 2**maxElement
    return mean[maxElement], meanErrors[maxElement], stds[maxElement], corrSize

def binningAnalysisTriple(data1, data2, data3, printAlreadyConvergedWarning=False):
    #algo find max of errors -> use this binsize
    #if not converging -> max is last element then report error converging
    #here Triple means it uses the same binsize for three data sets, the max size is taken

    N = data1.shape[0]
    if N != data2.shape[0] or N != data3.shape[0]:
        print("binningAnalysisTriple: N != data2.shape[0] or N != data3.shape[0]")
    maxBinningSteps = int(np.log2(N) + 0) #so last binning has 2 elements, 1 element is useless for error

    mean1       = np.zeros(maxBinningSteps)
    meanErrors1 = np.zeros(maxBinningSteps)
    stds1       = np.zeros(maxBinningSteps)

    mean2       = np.zeros(maxBinningSteps)
    meanErrors2 = np.zeros(maxBinningSteps)
    stds2       = np.zeros(maxBinningSteps)

    mean3       = np.zeros(maxBinningSteps)
    meanErrors3 = np.zeros(maxBinningSteps)
    stds3       = np.zeros(maxBinningSteps)

    #starting data
    binnedData1    = data1
    binnedData2    = data2
    binnedData3    = data3

    mean1[0]       = np.mean(data1)
    meanErrors1[0] = np.std(data1) / np.sqrt(N)
    stds1[0]       = np.std(data1)

    mean2[0]       = np.mean(data2)
    meanErrors2[0] = np.std(data2) / np.sqrt(N)
    stds2[0]       = np.std(data2)

    mean3[0]       = np.mean(data3)
    meanErrors3[0] = np.std(data3) / np.sqrt(N)
    stds3[0]       = np.std(data3)

    #binning  up to maxBinningSteps
    for i in range(1, maxBinningSteps):
        #binsize = 2**i

        #binning step
        binnedData1 = reduceIntoBin2(binnedData1)
        binnedData2 = reduceIntoBin2(binnedData2)
        binnedData3 = reduceIntoBin2(binnedData3)
        N = binnedData1.shape[0] 

        #new error, mean
        mean1[i]       = np.mean(binnedData1)
        meanErrors1[i] = np.std(binnedData1) / np.sqrt(N)
        stds1[i]       = np.std(binnedData1)

        mean2[i]       = np.mean(binnedData2)
        meanErrors2[i] = np.std(binnedData2) / np.sqrt(N)
        stds2[i]       = np.std(binnedData2)

        mean3[i]       = np.mean(binnedData3)
        meanErrors3[i] = np.std(binnedData3) / np.sqrt(N)
        stds3[i]       = np.std(binnedData3)

    #take the binsize of the larger one
    #maxElement == 0 means the init data is used
    maxElement = max(np.argmax(meanErrors1), np.argmax(meanErrors2), np.argmax(meanErrors3))
    if (maxElement+1) == maxBinningSteps: 
        info = "   (elements in bin of max error: " + str(data1.shape[0] // 2**maxElement) + ")"
        print("binningAnalysisTriple: [NOT CONVERGED]  increase dataset," + info)
    if maxElement == 0 and printAlreadyConvergedWarning:
        info = "   (elements in bin of max error: " + str(data1.shape[0] // 2**maxElement) + ")"
        print("binningAnalysisTriple: [Already CONVERGED]  first error is largest," + info)
 
    #bin to requested size
    binnedData1 = data1
    binnedData2 = data2
    binnedData3 = data3
    for i in range(1, maxElement):
        binnedData1 = reduceIntoBin2(binnedData1)
        binnedData2 = reduceIntoBin2(binnedData2)
        binnedData3 = reduceIntoBin2(binnedData3)

    #return
    corrSize = 2**maxElement
    return mean1[maxElement], meanErrors1[maxElement], stds1[maxElement], mean2[maxElement], meanErrors2[maxElement], stds2[maxElement], mean3[maxElement], meanErrors3[maxElement], stds3[maxElement], corrSize, binnedData1, binnedData2, binnedData3

#--------------------------------------------------------------------

@jit(cache=True)
def binderCumulant(m2, m4):
    r2 = m4 / (m2**2 + 1e-15)
    return 1.5 * (1.0 - r2 / 3.0)

@jit(cache=True)
def mBinderCuJackknife(binnedMag2, binnedMag4):
    #for binderCumulant after binning

    #init U0
    size = binnedMag2.shape[0] 
    arr  = np.zeros(size) 
    for i in range(size):
        arr[i] = binderCumulant(binnedMag2[i], binnedMag4[i])
    U0 = np.mean(arr)

    #cutting
    cutArr = np.zeros(size - 1)
    U      = np.zeros(size)

    for skip in range(size):    
        index = 0
        for i in range(size):
            if i == skip:
                continue
            
            cutArr[index] = binderCumulant(binnedMag2[i], binnedMag4[i])
            index += 1
        
        U[skip] = np.sum(cutArr) / (size - 1)
    
    Uq = np.mean(U)

    mean = U0 - (size - 1) * (Uq - U0)
    error = np.std(U) * np.sqrt(size-1)

    return mean, error

#--------------------------------------------------------------------

@jit(cache=True)
def magSusceptibility(mAbs, m2, N, T):
    return (N/T) * (m2 - mAbs**2)

@jit(cache=True)
def mSuscJackknife(binnedMagAbs, binnedMag2, N, T):
    #for mSuscBins after binning

    #init U0
    size = binnedMagAbs.shape[0] 
    arr  = np.zeros(size) 
    for i in range(size):
        arr[i] = magSusceptibility(binnedMagAbs[i], binnedMag2[i], N, T)
    U0 = np.mean(arr)

    #cutting
    cutArr = np.zeros(size - 1)
    U      = np.zeros(size)

    for skip in range(size):    
        index = 0
        for i in range(size):
            if i == skip:
                continue
            
            cutArr[index] = magSusceptibility(binnedMagAbs[i], binnedMag2[i], N, T)
            index += 1
        
        U[skip] = np.sum(cutArr) / (size - 1)
    
    Uq = np.mean(U)

    mean = U0 - (size - 1) * (Uq - U0)
    error = np.std(U) * np.sqrt(size-1)

    return mean, error

#--------------------------------------------------------------------

@jit(cache=True)
def k3(mAbs, m2, mAbs3, N, T):
    return (mAbs3 - 3.0 * m2 * mAbs + 2.0 * mAbs**3) * (N/T)

@jit(cache=True)
def mK3Jackknife(binnedMagAbs, binnedMag2, binnedMagAbs3, N, T):
    #for k3 after binning

    #init U0
    size = binnedMagAbs.shape[0] 
    arr  = np.zeros(size) 
    for i in range(size):
        arr[i] = k3(binnedMagAbs[i], binnedMag2[i], binnedMagAbs3[i], N, T)
    U0 = np.mean(arr)

    #cutting
    cutArr = np.zeros(size - 1)
    U      = np.zeros(size)

    for skip in range(size):    
        index = 0
        for i in range(size):
            if i == skip:
                continue
            
            cutArr[index] = k3(binnedMagAbs[i], binnedMag2[i], binnedMagAbs3[i], N, T)
            index += 1
        
        U[skip] = np.sum(cutArr) / (size - 1)
    
    Uq = np.mean(U)

    mean = U0 - (size - 1) * (Uq - U0)
    error = np.std(U) * np.sqrt(size-1)

    return mean, error

#--------------------------------------------------------------------

def calc_spin_spin_corr(state):
    corr = -1

    #calc <sigma>^2

    #for each r (max L):
    #calc corr
    #<si sj>

    #fit A*epx(-r/zeta)


    #do this for gan and mc data

    

    return corr