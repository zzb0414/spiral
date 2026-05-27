# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:33:09 2022

@author: yongquan.ye@united-imaging.com

This Template shows examplinary codes for reading the k-space and/or image data
reconstructed as .npy files
"""

import sys
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
plt.gray()
plt.rcParams['figure.dpi']= 1200

#%%
raw_fn = input('Input raw file here: ')
raw_fn = raw_fn.strip("'")
dn = os.path.dirname(raw_fn)
rawfn = os.path.basename(raw_fn)

substr = rawfn.split('_')
UID = substr[1]

raw_info_fn = 'UID_' + UID +'_info.npy'            
raw_info = np.load (dn +'/' + raw_info_fn, allow_pickle = True)

UID = raw_info[0].find('.//MeasUID/Value').text

#%% k-sapce data, iffted and cliped along RO dimensino
for ii in (range(np.size(raw_info[1]))): 
    fn_full = raw_info[1][ii]
    k_dn = os.path.dirname(fn_full)
    k_fn = os.path.basename(fn_full)    
    if os.path.exists(fn_full):        
        KDat = np.load(fn_full)
        # add your own processing codes here
    
    #get dimension info
    substr = k_fn[:-4].split('_')
    dim = substr[1].split('x')
    nRO, nPE, nSLC = int(dim[0]), int(dim[1]), int(dim[2])
    nCH = int(substr[2][3:])
    iEch = int(substr[3][3:])
    iSet = int(substr[4][3:])
    iRep = int(substr[5][3:])
    iAve = int(substr[6][3:])
    iUD0, iUD1, iUD2, iUD3, iUD4 = int(substr[8]), int(substr[9]), int(substr[10]), int(substr[11]), int(substr[12]), 
    print(str([nRO, nPE, nSLC, nCH, iEch, iSet, iRep, iAve, iUD0, iUD1, iUD2, iUD3, iUD4]))
    

#%%s PPA recon image data
for ii in (range(np.size(raw_info[1]))): 
    fn_full = raw_info[1][ii]
    k_dn = os.path.dirname(fn_full)
    k_fn = os.path.basename(fn_full)  
    i_fn = 'i' + k_fn[1:]
    recon_dn = os.path.dirname(k_dn) + '/UID_'+UID+'_I/'
    if os.path.exists(recon_dn + i_fn):        
        ImgDat = np.load(recon_dn + i_fn)    
        # add your own processing codes here
    
    #get dimension info
    substr = k_fn[:-4].split('_')
    dim = substr[1].split('x')
    nRO, nPE, nSLC = int(dim[0]), int(dim[1]), int(dim[2])
    nCH  = int(substr[2][3:])
    iEch = int(substr[3][3:])
    iSet = int(substr[4][3:])
    iRep = int(substr[5][3:])
    iAve = int(substr[6][3:])
    iUD0, iUD1, iUD2, iUD3, iUD4 = int(substr[8]), int(substr[9]), int(substr[10]), int(substr[11]), int(substr[12])
    print(str([nRO, nPE, nSLC, nCH, iEch, iSet, iRep, iAve, iUD0, iUD1, iUD2, iUD3, iUD4]))
 

#%% read in any single npy data
fn_full = input('Input raw recon npy file here: ')
fn_full = fn_full.strip("'")
Dat = np.load(fn_full)
fn = os.path.basename(fn_full)

if fn[0] == 'k': 
    dat2d = Dat[0,:,:,2]
    plt.imshow(np.abs(dat2d))
    plt.show()
elif fn[0] == 'i':
    dat2d = Dat[0,:,:,0]
    plt.imshow(np.abs(dat2d))
    plt.show()










