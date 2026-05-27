# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:55:53 2021

@author: yongquan.ye@united-imaging.com
"""

import numpy as np
import glob
from xml.etree import ElementTree as et

def ExtractProt(Rawfn):
    fid = open(Rawfn, 'rb')
    fid_temp = np.fromfile(fid,dtype='byte',count=1)
    if fid_temp==82:
         prot_shift = 12 #fseek(fid,12,'bof');
    else:
         prot_shift = 0 #fseek(fid,0,'bof');    
    
    
    ProtLen = np.fromfile(fid,offset=prot_shift-1,dtype='uint32',count=1)
    print('*** protocol length: ' + str(ProtLen[0]))
    
    prot =np.fromfile(fid,dtype='byte',count=ProtLen[0])
    fid.close()
    
    
    
    prot =prot.tobytes().decode("ascii", "ignore")
    prot_fn = Rawfn[:-3]+'prot'
    fid = open(prot_fn,'w')
    fid.write(prot)
    fid.close()    
    
    # if fid_temp==82:
    #     #DataStartOffset = prot_shift-1+ProtLen[0]+4+12
    #     DataStartOffset = prot_shift+ProtLen[0]+4
    # else:
    #     DataStartOffset = prot_shift+ProtLen[0]+4
    DataStartOffset = prot_shift+ProtLen[0]+4
    
    #change protocol values
    
       
    tree = et.parse(prot_fn)
    protname = tree.find('./Header/ProtName').text
    if 'PALAdvNorm' not in protname:        
        # tree.find('./Root/IRIP/PipeLineConfig/Procedure/Value').text = 'UmrIripDataOutputProcedure'
        # tree.find('./Root/IRIP/FromUI/Normalize/Value').text = '0'
        # tree.find('./Root/IRIP/FromUI/NoiseDecor/Value').text = '0'
        #tree.find('./Root/IRIP/FromUI/DefaultNoiseDecor/Value').text = '0'    
        try:
            ccomp=tree.find('.//CoilCompression/Value').text
            if ccomp=='true':
                tree.find('.//CoilCompression/Value').text = 'false'
                RxChID = tree.findall ('.//RxChannelID/Value')
                ChLen = []
                for ii in RxChID:
                    ii = ii.text.replace(';',' ')
                    MaxChID = [int(s) for s in ii.split() if s.isdigit()]
                    # print(MaxChID)
                    for jj in MaxChID:
                        ChLen.append(jj)
                ChLen = np.max(ChLen)+1 
                tree.find('.//ReconChannelNum/Value').text = str(ChLen)
        except:
            pass
       
        # coiltype=tree.find('.//CoilSelection/SelectedElementGroupInfo/ss0/ElementGroupName/Value').text

        tree.write(prot_fn)
    
    return DataStartOffset
    
    
        


        