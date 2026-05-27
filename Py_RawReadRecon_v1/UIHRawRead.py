# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:52:58 2021

@author: yongquan.ye@united-imaging.com
"""
import os
import numpy as np
from Py_RawReadRecon_v1 import ExtractProtFromRaw
from Py_RawReadRecon_v1 import ReadUIHDHL
from xml.etree import ElementTree as et
import shutil


# Need to include an export option
def Run(raw_fn, ScanMode='2D'):
    if ScanMode in ['2D', '3D', '3DRadial', '3DCones']:
        print('ScanMode: ', ScanMode)
    else:
        print('the current supprot ScanMode is: 2D, 3D, 3DRadial and 3DCones, the input ScanMode is: ', ScanMode)
        return False

    dn=os.path.dirname(raw_fn)    
    
    # Extract protocol and save as a file
    DataStartOffset = ExtractProtFromRaw.ExtractProt(raw_fn)
    
    # Extract prot param from prot file
    prot_fn = raw_fn[:-3]+'prot'
    prot_tree = et.parse(prot_fn)
    ROLen = np.int16(prot_tree.find('.//MatrixRO/Value').text) * 2
    bPartialEcho = np.int16(prot_tree.find('.//PartialEcho/Value').text)
    # PPAMode = np.int16(prot_tree.find('.//PPA/Method/Value').text)      
    SlcLen = np.int16(prot_tree.find('.//NumberOfSlice/Value').text)
    EchLen = np.int16(prot_tree.find('.//Contrast/Value').text)
    AveLen = np.int16(prot_tree.find('.//Average/Value').text)
    RepLen = np.int16(prot_tree.find('.//Repetition/Value').text)
    SetLen = np.int16(prot_tree.find('.//ExDim/SET/Value').text)
    UD0Len = np.int16(prot_tree.find('.//ExDim/UD0/Value').text)
    UD1Len = np.int16(prot_tree.find('.//ExDim/UD1/Value').text)
    UD2Len = np.int16(prot_tree.find('.//ExDim/UD2/Value').text)
    UD3Len = np.int16(prot_tree.find('.//ExDim/UD3/Value').text)
    UD4Len = np.int16(prot_tree.find('.//ExDim/UD4/Value').text)
    UID = prot_tree.find('.//MeasUID/Value').text    
    Dim = np.int16(prot_tree.find('.//Dimension/Value').text)
        
    raw_info_fn = 'UID_' + UID +'_info.npy'  
    if os.path.exists(dn+'/' + raw_info_fn):
        raw_info = np.load(dn + '/' + raw_info_fn, allow_pickle=True)
        return raw_info
        
    # Get actual data dimensions and lengths    
    print('Data estimating, please wait....')
    fid = open(raw_fn,'rb')
    CurOffSet = DataStartOffset    
    dim_lists = np.zeros([EchLen, SetLen, RepLen, AveLen, UD0Len, UD1Len, UD2Len, UD3Len, UD4Len])
    PELen = 0
    ChLen = 0
    SPELen = 0
    SegLen = 0
    cnt = 0
    while(1):
        # print(cnt)
        
        # Read DHL header
        DHLDat=np.fromfile(fid, dtype='byte', count=192, offset = CurOffSet) #the header is 192 byte
        
        dhl = ReadUIHDHL.UIHDHL()
        dhl.SetDHLDat(DHLDat)
        dhlinfo = dhl.ReadUIHDHL()
        
        bIsAcqEnd = dhlinfo.DHLFlag('DHL_ACQUISITION_END')    
        if bIsAcqEnd:
            break
       
        # Read data line
        DatLineLength = dhlinfo.Channels*(dhlinfo.Samples+2)*2    
        CurOffSet = DatLineLength * 4
        
        # Skip noise scan
        bIsImagingLine = not dhlinfo.DHLFlag('DHL_NOISE_SCAN')
        bIsImagingLine &= not dhlinfo.DHLFlag('DHL_FEEDBACK')
        bIsImagingLine &= not dhlinfo.DHLFlag('DHL_PHASE_CORRECTION')
        bIsImagingLine &=  not dhlinfo.DHLFlag('DHL_USER_0')
        bIsImagingLine &=  not dhlinfo.DHLFlag('DHL_USER_1')
        bIsImagingLine &=  not dhlinfo.DHLFlag('DHL_USER_2')
        bIsImagingLine &=  not dhlinfo.DHLFlag('DHL_USER_3')
        bIsImagingLine &=  not dhlinfo.DHLFlag('DHL_USER_4')
        if not bIsImagingLine:
            continue
        cnt += 1
        
        # By zz
        ROLen = max(ROLen, (dhlinfo.Samples)) # Need to consider rawdata version
        if ScanMode == '3DCones':
            PELen = max(PELen, dhlinfo.cShot)
        else:
            PELen = max(PELen, dhlinfo.cPEInd)
            if cnt == 1:
                PEInd_first = PELen
        
        ChLen  = max(ChLen, dhlinfo.Channels)
        SPELen = max(SPELen, dhlinfo.cSPEInd)
        SlcLen = max(SlcLen, dhlinfo.cSlice)
        SegLen = max(SegLen, dhlinfo.cSeg)

        cRep =  dhlinfo.cRepeat
        cAve =  dhlinfo.cAverage     
        cSet =  dhlinfo.cSet
        cEch =  dhlinfo.cContrast
        cUD0 =  dhlinfo.User[0]
        cUD1 =  dhlinfo.User[1]
        cUD2 =  dhlinfo.User[2]
        cUD3 =  dhlinfo.User[3]
        cUD4 =  dhlinfo.User[4] 
        
        dim_lists[cEch, cSet, cRep, cAve, cUD0, cUD1, cUD2, cUD3, cUD4] = 1
        
    fid.close()              
    
    # Create dict for data volumes
    dim_lists_flat=[]
    for Ech in range(EchLen):
        for Set in range(SetLen):
            for Rep in range(RepLen):
                for Ave in range(AveLen):
                    for UD0 in range (UD0Len):
                        for UD1 in range (UD1Len):
                            for UD2 in range (UD2Len):
                                for UD3 in range (UD3Len):
                                    for UD4 in range (UD4Len):
                                        if dim_lists[Ech, Set, Rep, Ave, UD0, UD1, UD2, UD3, UD4]==1:
                                            dim_lists_flat.append([Ech, Set, Rep, Ave, UD0, UD1, UD2, UD3, UD4])
    
    PELen = PELen + 1 - PEInd_first
    SPELen = SPELen + 1
    SegLen = SegLen + 1
    TotalVolNum = np.shape(dim_lists_flat)[0]
    if ScanMode == '2D':
        SPELen = SlcLen 
    if ScanMode == '3DCones':
        SPELen = 1   
    
    DatVol = dict()
    for ii in range (TotalVolNum):
        DatVol[ii] = np.zeros([ChLen, ROLen, PELen, SPELen, SegLen], dtype='complex64')

    if ScanMode == '3DRadial':
        rotAngle = np.zeros([PELen, SPELen], dtype='double')
    
    # Read data
    BinFN=[] 
         
    # Start reading data            
    fid = open(raw_fn,'rb')
    np.fromfile(fid, dtype='byte', count=DataStartOffset) # Skip the protocol info    
    
    print('Reading data, please wait....')
    while(1):
        #read DHL header
        DHLDat = np.fromfile(fid,dtype='byte',count=192)
        
        dhl = ReadUIHDHL.UIHDHL()
        dhl.SetDHLDat(DHLDat)   
        dhlinfo = dhl.ReadUIHDHL()     
        
        bIsAcqEnd = dhlinfo.DHLFlag('DHL_ACQUISITION_END')    
        if bIsAcqEnd:
            break
       
        # Read data line
        DatLineLength = dhlinfo.Channels * (dhlinfo.Samples + 2) * 2
        LDat = np.fromfile(fid, dtype='float32', count=DatLineLength)

        # Skip noise scan
        bIsImagingLine = not dhlinfo.DHLFlag('DHL_NOISE_SCAN')
        bIsImagingLine &= not dhlinfo.DHLFlag('DHL_FEEDBACK')
        bIsImagingLine &= not dhlinfo.DHLFlag('DHL_PHASE_CORRECTION')
        bIsImagingLine &=  not dhlinfo.DHLFlag('DHL_USER_0')
        bIsImagingLine &=  not dhlinfo.DHLFlag('DHL_USER_1')
        bIsImagingLine &=  not dhlinfo.DHLFlag('DHL_USER_2')
        bIsImagingLine &=  not dhlinfo.DHLFlag('DHL_USER_3')
        bIsImagingLine &=  not dhlinfo.DHLFlag('DHL_USER_4')
        if not bIsImagingLine:
            continue
      
        # Read DHL tags
        cPEInd = dhlinfo.cPEInd
        if ScanMode in ['3D', '3DRadial']:
            cSPEInd = dhlinfo.cSPEInd
        elif ScanMode == '3DCones':
            cSPEInd = 0
            cPEInd = dhlinfo.cShot
        else:
            cSPEInd = dhlinfo.cSlice
        cSegInd = dhlinfo.cSeg
        
        cRep =  dhlinfo.cRepeat
        cAve =  dhlinfo.cAverage     
        cSet =  dhlinfo.cSet
        cEch =  dhlinfo.cContrast
        cUD0 =  dhlinfo.User[0]
        cUD1 =  dhlinfo.User[1]
        cUD2 =  dhlinfo.User[2]
        cUD3 =  dhlinfo.User[3]
        cUD4 =  dhlinfo.User[4]    
        
        ctags = [cEch, cSet, cRep, cAve, cUD0, cUD1, cUD2, cUD3, cUD4]
        VolInd = -1
        for ii in range(TotalVolNum):
            if ctags==dim_lists_flat[ii]:
                VolInd = ii
                break
                
        if ScanMode == '3DRadial':
            rotAngle[cPEInd, cSPEInd] = dhlinfo.IRIPdata

        if VolInd!=-1:   
            if dhlinfo.Channels == ChLen:    
                LDatCplx = LDat[0::2]+1j*LDat[1::2]
                LDatCplx = np.reshape(LDatCplx,(ChLen,dhlinfo.Samples+2))
                LDatCplx = LDatCplx[:,2:]  #remove the first two points for rawdata version >=2            
                if dhlinfo.DHLFlag('DHL_READOUT_REVERSION'):
                    LDatCplx = np.flip(LDatCplx,axis = 1)                                        
                    
                ActualROLen = np.shape(LDatCplx)[1]
                if bPartialEcho==0:
                    DatVol[VolInd][:, :, cPEInd-PEInd_first, cSPEInd, cSegInd] = LDatCplx
                else:
                    DatVol[VolInd][:, ROLen-ActualROLen:ROLen, cPEInd-PEInd_first, cSPEInd, cSegInd] = LDatCplx
                                                   
    fid.close()
    
    # print([PELen,SPELen])
    print(f"rawdata shape: {DatVol[0].shape}")

    # Save data      
    for VolInd in (range (TotalVolNum)):                    
        ctags = dim_lists_flat[VolInd]
        npy_fn = 'k_' + str(int(ROLen)) + 'x' + str(PELen) + 'x' + str(SPELen) + 'x' + str(SegLen) +'_CHA' + str(ChLen) + \
                '_Ech' + str(ctags[0]) + '_Set'+ str(ctags[1]) +'_Rep'+str(ctags[2])+'_Ave' + str(ctags[3]) + \
                '_UD_' + str(ctags[4]) +'_' + str(ctags[5]) +'_' + str(ctags[6]) +'_' + str(ctags[7]) +'_' + str(ctags[8]) + '.npy'            
        print(f"Saving data for {npy_fn}")

        DatVolTemp = np.copy(DatVol[VolInd])
        # save npy file, CHxROxPExSPE
        # saving to C: then copy to destination is faster in the following way
        # if the destination is a network folder
        kspace_dn = dn + '/UID_' + UID + '_K'
        if not os.path.isdir(kspace_dn):
            os.mkdir(kspace_dn) 
        np.save(kspace_dn + '/' + npy_fn, DatVolTemp.astype('complex64'))  
        BinFN.append(kspace_dn + '/' + npy_fn)  
        print(f"data size: {DatVolTemp.shape}")

    if ScanMode == '3DRadial':
        np.save(kspace_dn + '/rotAngle.npy', rotAngle)

    # Save info
    raw_info_fn = 'UID_' + UID +'_info.npy'            
    raw_info = [BinFN]
    np.save(kspace_dn +'/' + raw_info_fn, raw_info) 
    
    return raw_info


