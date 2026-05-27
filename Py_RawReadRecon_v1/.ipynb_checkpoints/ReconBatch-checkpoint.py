# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:41:59 2021

@author: yongquan.ye@united-imaging.com
"""
# import sys
# sys.modules[__name__].__dict__.clear()
import shutil
import os
import UIHRawRead
import numpy as np
# import matplotlib.pyplot as plt
# plt.gray()
# plt.rcParams['figure.dpi']= 1200
import multiprocessing as mp
from UIHPPARecon import UIHPPARecon
import SaveDicom
import time
# import PhaseCoilCombine
# import PhaseCoilCombine_VRC

    
def main(raw_fn):
    if os.path.isdir(raw_fn):    
        # extract and modify protocol for offline recon
        datfolder = raw_fn
        for path, subdirs, files in os.walk(datfolder):           
            for name in files:
                if name.endswith('.raw'):
                    print(path + '/' +name)
                    # Saves the RO-ffted, RO-cliped k-space data
                    raw_info = UIHRawRead.Run(path + '/' +name)  
                    
                    # uncomment this if you only need the k-space data
                    # continue
                    
                    #perform ppa recon, or insert your own recon codes
                    PPARecon(raw_info)
        
    else:
        if not raw_fn.endswith('.raw'):
            print('This is not a .raw file, existing...')
            # sys.exit()        
        else:
            # Saves the RO-ffted, RO-cliped k space data
            raw_info=UIHRawRead.Run(raw_fn)
            
            # uncomment this if you only need the k-space data
            # sys.exit()
            
            #perform ppa recon, or insert your own recon codes
            PPARecon(raw_info)
 
# PPArecon
def PPARecon(raw_info):
    processnum = 8#mp.cpu_count()
    mp.Pool(processnum)
    for ii in (range(np.size(raw_info[1]))):    
        #read bin data, CHxROxPExSPE5
        fn_full = raw_info[1][ii]
        dn = os.path.dirname(fn_full)
        fn = os.path.basename(fn_full)    
        img_fn = 'i' + fn[1:]
            
        UID = raw_info[0].find('.//MeasUID/Value').text
        recon_dn = os.path.dirname(dn) + '/UID_'+UID+'_I/'
        if not os.path.exists(recon_dn + img_fn):
            print('Performing PPA recon on volume '+ str(ii) + '/' + str(np.size(raw_info[1])))
            
            dat = np.load(fn_full)    
            dim = np.shape(dat)
            
            Dim = np.int16(raw_info[0].find('.//Dimension/Value').text)
            if Dim==1:
                ScanMode = '2D'   #CHxROxPExSLC    
            else:
                ScanMode = '3D'   #CHxROxPExSPE
            
            PPAMode = np.int16(raw_info[0].find('.//PPA/Method/Value').text)
            if PPAMode == 0:# No PPA
                if ScanMode == '3D' :      
                    dat_recon  = np.complex64(np.fft.ifft(dat,axis=2))
                    dat_recon  = np.complex64(np.fft.ifft(dat_recon,axis=3))
                    dat_recon  = np.complex64(np.fft.ifftshift(dat_recon,axes=(2,3)))   
                else:
                    dat_recon  = np.complex64(np.fft.ifft(dat,axis=2))
                    dat_recon  = np.complex64(np.fft.ifftshift(dat_recon,axes=(2,)))   
            
            elif PPAMode == 1: # normal PPA
                ppa = UIHPPARecon(raw_info[0])    
                # complex image        
                manager = mp.Manager()
                return_dict = manager.dict()
                
                t1 = time.time()
                if  ScanMode == '3D' :
                    seg = processnum
                    seglength = dim[1]/seg
                else:
                    seg = dim[3]
                    seglength = 1
                jobs = []
                for iseg in range (seg):      
                    if  ScanMode == '3D' :
                        p = mp.Process(target=ppa.Run, args=(dat[:,int(iseg*seglength):int((iseg+1)*seglength),:,:],iseg, return_dict))
                    else:
                        p = mp.Process(target=ppa.Run, args=(dat[:,:,:,iseg],iseg, return_dict))
                    jobs.append(p)
                    p.start()
                
                for proc in jobs:
                    proc.join()
                
                dat_recon = np.zeros_like(dat,dtype = 'complex64')
                for iseg in range (seg):
                    if  ScanMode == '3D' :
                        dat_recon[:,int(iseg*seglength):int((iseg+1)*seglength),:,:] = return_dict.get(iseg)
                    else:
                        dat_recon[:,:,:,iseg] = return_dict.get(iseg)                
                
                t2 = time.time()
                print('Recon time is ' + str(t2 - t1) + 's')
            
            else: # for other acceleration, not yet supported
                print('Unsupported PPA mode, existing...')
                pass
                # sys.exit()                
                    
            #=========================================================   
            # save raw image data    
            if not os.path.isdir(recon_dn):
                os.mkdir(recon_dn)    
            np.save ('C:/'+ img_fn , dat_recon)
            shutil.move('C:/' + img_fn, recon_dn + img_fn)
        
        raw_info_fn = 'UID_' + UID +'_info.npy'  
        shutil.copy(dn + '/' + raw_info_fn , recon_dn + raw_info_fn)
            
        if 'dat_recon' not in locals():
            dat_recon = np.load (recon_dn + img_fn)  
        dim = np.shape(dat_recon)
            
        
        # save magnitude
        dcm_dn = os.path.dirname(dn) + '/UID_' + UID + '_dcm/'
        if not os.path.isdir(dcm_dn):
            os.mkdir(dcm_dn)         
        sos = np.sqrt(np.sum(np.square(np.abs(dat_recon)),axis=0))
        sos = np.ascontiguousarray(sos)        
        pos = fn.find('Ech')
        SaveDicom.save(sos,raw_info[0], dcm_dn + fn[pos:-4], 'M',100 + ii,UID)                
    
    print('AllDone!!!')



if __name__ == '__main__':
    raw_fn = input('Input raw file or folder here: ')
    raw_fn = raw_fn.strip("'")

    main(raw_fn)    
    