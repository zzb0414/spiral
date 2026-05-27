import os
import numpy as np
import matplotlib.pyplot as plt
#import math

#fileBase = "D:\\UIMRIS\\BRANCHES\\uMRV10_3T\\UIH\\appdata\\MRSiteData\\Share\\Service\\Data\\Grad_Response"
fileBase = "C:/Users/zhibo.zhu/OneDrive - UIH Group/Documents/MATLAB/ssfse_spiral/Grad_Response_Spi"
#fileBase = "F:\\790-system-data\\Grad_Response-bk-1116-2023"

filepath = fileBase + "\\Fre_Axis.dat"
fFreq = np.fromfile(filepath, dtype='float32')
arrLength = fFreq.size

filepath = fileBase + "\\Grad_Response_X.dat"
fXGradFunc_raw = np.fromfile(filepath, dtype='float32') #the first half is real, the second half is imag
fXgradFunc = fXGradFunc_raw[0:arrLength] + fXGradFunc_raw[arrLength:] * 1j

filepath = fileBase + "\\Grad_Response_Y.dat"
fYGradFunc_raw = np.fromfile(filepath, dtype='float32') #the first half is real, the second half is imag
fYgradFunc = fYGradFunc_raw[0:arrLength] + fYGradFunc_raw[arrLength:] * 1j

filepath = fileBase + "\\Grad_Response_Z.dat"
fZGradFunc_raw = np.fromfile(filepath, dtype='float32') #the first half is real, the second half is imag
fZgradFunc = fZGradFunc_raw[0:arrLength] + fZGradFunc_raw[arrLength:] * 1j

print(fFreq.shape)

#plt.plot(np.log10(fFreq), np.abs(fXgradFunc))

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

#fileBase2 = "D:\\UIMRIS\\BRANCHES\\uMRV10_3T\\UIH\\appdata\\MRSiteData\\Share\\Service\\Data\\Grad_Response_790_update_ECC_0717_2023"
#fileBase2 = "D:\\UIMRIS\\BRANCHES\\uMRV10_3T\\UIH\\appdata\\MRSiteData\\Share\\Service\\Data\\Grad_Response_790_07072013"
fileBase2 = "C:/Users/zhibo.zhu/OneDrive - UIH Group/Documents/MATLAB/ssfse_spiral/Grad_Response_Spi"

filepath = fileBase2 + "\\Fre_Axis.dat"
fFreq2 = np.fromfile(filepath, dtype='float32')
arrLength2 = fFreq2.size

filepath = fileBase2 + "\\Grad_Response_X.dat"
fXGradFunc_raw = np.fromfile(filepath, dtype='float32') #the first half is real, the second half is imag
fXgradFunc2 = fXGradFunc_raw[0:arrLength] + fXGradFunc_raw[arrLength:] * 1j

filepath = fileBase2 + "\\Grad_Response_Y.dat"
fYGradFunc_raw = np.fromfile(filepath, dtype='float32') #the first half is real, the second half is imag
fYgradFunc2 = fYGradFunc_raw[0:arrLength] + fYGradFunc_raw[arrLength:] * 1j

filepath = fileBase2 + "\\Grad_Response_Z.dat"
fZGradFunc_raw2 = np.fromfile(filepath, dtype='float32') #the first half is real, the second half is imag
fZgradFunc2 = fZGradFunc_raw2[0:arrLength] + fZGradFunc_raw2[arrLength:] * 1j

diff_fZ = fZgradFunc2 - fZgradFunc

line_type = '-'
plt.figure(1)
plt.subplot(311)
plt.plot(fFreq, np.abs(fXgradFunc), line_type, label='default-ECC')
plt.plot(fFreq, np.abs(fXgradFunc2), line_type, label='shorter-ECC')
plt.legend()
plt.title('X-axis')

plt.subplot(312)
plt.plot(fFreq, np.abs(fYgradFunc), line_type, label='default-ECC')
plt.plot(fFreq, np.abs(fYgradFunc2), line_type, label='shorter-ECC')
plt.legend()
plt.title('Y-axis')

plt.subplot(313)
plt.plot(fFreq, np.abs(fZgradFunc), line_type, label='default-ECC')
plt.plot(fFreq, np.abs(fZgradFunc2), line_type, label='shorter-ECC')
plt.legend()
plt.title('Z-axis')

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
plt.xlabel('BW (Hz)', fontdict=font)

plt.show()

