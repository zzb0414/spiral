# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 19:35:30 2021

@author: yongquan.ye@united-imaging.com
"""

import os
import pydicom
import numpy as np
from xml.etree import ElementTree as et
from  shutil import rmtree

def save(DicomDat,ProtTree,DirName,SeriesDescription,SN,UID):
    dim=np.shape(DicomDat) # ROxPExSPE    
    
    DcmDirName = DirName + '/' + SeriesDescription+'/'
    if os.path.exists(DcmDirName):
        rmtree(DcmDirName)
    os.makedirs(DcmDirName)
    
    if UID:   
        SeriesUID = UID
        StudyUID = UID
        FrameUID = UID
    else:
        SeriesUID = pydicom.uid.generate_uid()
        StudyUID = pydicom.uid.generate_uid()
        FrameUID = pydicom.uid.generate_uid()
        
    for ispe in range(dim[2]):
        meta = pydicom.Dataset()
        meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian          
        
        ds = pydicom.dataset.Dataset()
        ds.file_meta = meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
        ds.PatientName = "Anon"
        ds.PatientID = "123456"
        
        ds.Modality = "MR"
        ds.SeriesInstanceUID = SeriesUID
        ds.StudyInstanceUID = StudyUID
        ds.FrameOfReferenceUID = FrameUID
        ds.SeriesNumber = str(SN)
        
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SamplesPerPixel = 1
        ds.HighBit = 15
        
        ds.ImagesInAcquisition = "1"
        
        ds.Rows = dim[0]
        ds.Columns = dim[1]
        ds.InstanceNumber = ispe
        ds.SeriesDescription = SeriesDescription
        
        ds.ImagePositionPatient = r"0\0\1"
        ds.ImageOrientationPatient = r"1\0\0\0\-1\0"
        ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
        
        ds.RescaleIntercept = "0"
        ds.RescaleSlope = "1"
        ds.PixelSpacing = r"1\1"
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 1
        
        pydicom.dataset.validate_file_meta(file_meta=meta, enforce_standard=True)
        
        ds.PixelData = DicomDat[:,:,ispe].astype('int16')
        
        dicomfn = DirName + '/' + SeriesDescription+'/'+ str(ispe) + '.dcm'
        ds.save_as(dicomfn)