import os
import numpy as np
import matplotlib.pyplot as plt
#import math

folder_name = "C:/Users/zhibo.zhu/OneDrive - UIH Group/Documents/Jupyter/spiral/Grad_Response_Spi/Omega"

file_path = folder_name + "\\Fre_Axis.dat"
freq_axis = np.fromfile(file_path, dtype='float32')
n = freq_axis.size

file_path = folder_name + "\\B0_Response_X.dat"
Hx_B0 = np.fromfile(file_path, dtype='float32') #the first half is real, the second half is imag
Hx_B0 = Hx_B0[0:n] + Hx_B0[n:] * 1j

file_path = folder_name + "\\B0_Response_Y.dat"
Hy_B0 = np.fromfile(file_path, dtype='float32') #the first half is real, the second half is imag
Hy_B0 = Hy_B0[0:n] + Hy_B0[n:] * 1j

file_path = folder_name + "\\B0_Response_Z.dat"
Hz_B0 = np.fromfile(file_path, dtype='float32') #the first half is real, the second half is imag
Hz_B0 = Hz_B0[0:n] + Hz_B0[n:] * 1j

file_path = folder_name + "\\Grad_Response_X.dat"
Hx_grad = np.fromfile(file_path, dtype='float32') #the first half is real, the second half is imag
Hx_grad = Hx_grad[0:n] + Hx_grad[n:] * 1j

file_path = folder_name + "\\Grad_Response_Y.dat"
Hy_grad = np.fromfile(file_path, dtype='float32') #the first half is real, the second half is imag
Hy_grad = Hy_grad[0:n] + Hy_grad[n:] * 1j

file_path = folder_name + "\\Grad_Response_Z.dat"
Hz_grad = np.fromfile(file_path, dtype='float32') #the first half is real, the second half is imag
Hz_grad = Hz_grad[0:n] + Hz_grad[n:] * 1j

Hfun = np.vstack((Hx_B0, Hy_B0, Hz_B0, Hx_grad, Hy_grad, Hz_grad))
print("Saving data ...")
np.save("freq_axis.npy", freq_axis.astype('float32'))
np.save("Hfun.npy", Hfun.astype('complex64'))