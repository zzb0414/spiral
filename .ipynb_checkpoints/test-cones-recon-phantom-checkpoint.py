import os
import numpy as np
import matplotlib.pyplot as plt

root_path = 'E:\\data\\rawdata-phantom-0415-2024-forDL\\'

full_file_name = root_path + "spiral-girf-ktraj\\ktraj_x.bin"

size = os.path.getsize(full_file_name)

crds_x = np.fromfile(full_file_name, dtype='float32', count=size//4)

full_file_name = root_path + "spiral-girf-ktraj\\ktraj_y.bin"
crds_y = np.fromfile(full_file_name, dtype='float32', count=size//4)

full_file_name = root_path + "spiral-girf-ktraj\\ktraj_z.bin"
crds_z = np.fromfile(full_file_name, dtype='float32', count=size//4)

full_file_name = root_path + "spiral-girf-ktraj\\density_cones.bin"
dens = np.fromfile(full_file_name, dtype='float32', count=size//4)


full_file_name = root_path + 'rawdata_cones_all.bin'
#ksp_finufft = np.reshape(ksp, [num_coils, -1])

num_coils = 11
size = os.path.getsize(full_file_name)
iLength = size // num_coils // 8

ksp_finufft = np.fromfile(full_file_name, dtype='complex64', count=size//8)
ksp_finufft = np.reshape(ksp_finufft, [num_coils, iLength])

img_shape = [256,256,60]

import finufft
import time
tic = time.perf_counter()

f = np.zeros(img_shape)
for i in range(num_coils):
    print("current process coil: ", i)
    #cur_ksp = np.squeeze(np.reshape(ksp2[i, 10:, :], (1, -1))) * dens
    cur_ksp = np.squeeze(ksp_finufft[i, :])
    #img_t = finufft.nufft3d1(crds_x*2*np.pi, crds_y*2*np.pi, crds_z*2*np.pi, cur_ksp, (img_shape[0], img_shape[1], img_shape[2]))
    img_t = finufft.nufft3d1(crds_x, crds_y, crds_z, cur_ksp, (img_shape[0], img_shape[1], img_shape[2]))
    f += np.abs(img_t)**2
    
f = f**0.5
    
toc = time.perf_counter()
print(f"recon using: {toc - tic:0.4f} seconds") 

img2show = np.squeeze(f)#np.transpose(np.squeeze(img), [1, 2, 0])
from utils.imdisp import imdisp

imdisp(np.flip(np.transpose(np.abs(img2show[:,:,36:38]), [0,1, 2]), axis=0))