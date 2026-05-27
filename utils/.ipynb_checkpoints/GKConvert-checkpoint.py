import numpy as np

gamma_proton = 4257.59 #unit: Hz/Gauss
GRAD_SAMPLE_TIME_S = 1e-5 #unit: s

#RES, unit: cm
#inputG: unit: gauss/cm
#outputK: normalized to -Kmax/2 to Kmax/2
def GKConvert(inputG, RES):
    size = inputG.size
    outputK = np.zeros(size)
    tmpFactor = GRAD_SAMPLE_TIME_S * gamma_proton * 2*np.pi * RES    
    for i in range(size):
        if i == 0:
            outputK[0] = inputG[0] * tmpFactor
        else:
            outputK[i] = outputK[i-1] + (inputG[i]+inputG[i-1]) / 2 * tmpFactor

    return outputK


