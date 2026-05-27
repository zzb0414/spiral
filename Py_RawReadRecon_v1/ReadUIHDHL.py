# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 15:24:01 2021

@author: yongquan.ye@united-imaging.com
"""
import numpy as np

class UIHDHL:
    def __init__(self):
        self.Version=0
        self.ReceiveGain=0
        self.RFSettingInd=0
        self.DataLength=0
        self.MeasUID=0
        self.SysTimeStamp=0
        self.ScanTimeStamp=0
        self.VSMTimeStamp=0
        self.TabPositionX=0
        self.TabPositionY=0
        self.TabPositionZ=0
        self.Samples=0
        self.Channels=0
        self.CtrlFlags=0
        self.cRepeat=0
        self.cCardPhase=0
        self.cSlice=0
        self.cAverage=0
        self.cContrast=0
        self.cSet=0
        self.cShot=0
        self.cSeg=0
        self.cPEInd=0
        self.cSPEInd=0
        self.UserCtrlFlag=0
        self.User=[0,0,0,0,0]
        self.CutOffHead=0
        self.CutOffTail=0
        self.ROCenter=0
        self.PECenter=0
        self.SPECenter=0
        self.SlcPosition=0
        self.SlcOrientation=0
            
    def SetDHLDat(self,DHLDat):
        self.DHLDat=DHLDat

    def disp(self):
        for attr, val in vars(self).items():
            print(f"{attr}: {val}")
            
    def ReadUIHDHL(self):
        self.Version = np.uint8(self.DHLDat[0])
        self.ReceiveGain = np.uint8(self.DHLDat[2])
        self.DataLength = np.uint32(int.from_bytes(self.DHLDat[4:8],byteorder='little'))
        # self.MeasUID = np.uint64(int.from_bytes(self.DHLDat[8:16],byteorder='little'))
        # self.SysTimeStamp = np.fromstring(self.DHLDat,dtype='uint32',count=1)
        # self.ScanTimeStamp = np.fromstring(self.DHLDat,dtype='uint32',count=1)
        # self.VSMTimeStamp = np.fromstring(self.DHLDat,dtype='uint32',count=1)
        # self.TabPositionX = np.fromstring(self.DHLDat,dtype='int32',count=1)
        # self.TabPositionY = np.fromstring(self.DHLDat,dtype='int32',count=1)
        # self.TabPositionZ = np.fromstring(self.DHLDat,dtype='int32',count=1)
        self.Samples    =   np.uint32(int.from_bytes(self.DHLDat[40:42],byteorder='little'))
        self.Channels   =   np.uint32(int.from_bytes(self.DHLDat[42:44],byteorder='little'))
        # np.fromstring(self.DHLDat,dtype='uint8',count=4)  #skip 4
        self.CtrlFlags  =   self.DHLDat[48:56]
        self.cRepeat    =   int.from_bytes(self.DHLDat[56:58],byteorder='little')
        self.cCardPhase =   int.from_bytes(self.DHLDat[58:60],byteorder='little')
        self.cSlice     =   int.from_bytes(self.DHLDat[60:62],byteorder='little')
        self.cAverage   =   int.from_bytes(self.DHLDat[62:64],byteorder='little')
        self.cContrast  =   int.from_bytes(self.DHLDat[64:66],byteorder='little')
        self.cSet       =   int.from_bytes(self.DHLDat[66:68],byteorder='little')
        self.cShot      =   int.from_bytes(self.DHLDat[68:70],byteorder='little')
        self.cSeg       =   int.from_bytes(self.DHLDat[70:72],byteorder='little')
        self.cPEInd     =   int.from_bytes(self.DHLDat[72:74],byteorder='little')
        self.cSPEInd    =   int.from_bytes(self.DHLDat[74:76],byteorder='little')
        self.UserCtrlFlag = self.DHLDat[76:78]
        self.User[0]    =   np.uint32(int.from_bytes(self.DHLDat[78:80],byteorder='little'))
        self.User[1]    =   np.uint32(int.from_bytes(self.DHLDat[80:82],byteorder='little'))
        self.User[2]    =   np.uint32(int.from_bytes(self.DHLDat[82:84],byteorder='little'))
        self.User[3]    =   np.uint32(int.from_bytes(self.DHLDat[84:86],byteorder='little'))
        self.User[4]    =   np.uint32(int.from_bytes(self.DHLDat[86:88],byteorder='little'))
        # self.CutOffHead = np.fromstring(self.DHLDat,dtype='uint16',count=1)
        # self.CutOffTail = np.fromstring(self.DHLDat,dtype='uint16',count=1)
        self.ROCenter   =   int.from_bytes(self.DHLDat[92:94],byteorder='little')
        self.PECenter   =   int.from_bytes(self.DHLDat[94:96],byteorder='little')
        self.SPECenter  =   int.from_bytes(self.DHLDat[96:98],byteorder='little')
        
        return self
    
    def DHLFlag(self,tag):
        flag = np.uint64(int.from_bytes(self.CtrlFlags,'little'))
        flag_bin = np.binary_repr(flag,64)
        if tag == 'DHL_ACQUISITION_END':
            shift_bit=0
        elif tag == 'DHL_PHASE_CORRECTION':
            shift_bit=7
        elif tag == 'DHL_READOUT_REVERSION':
            shift_bit=8            
        elif tag == 'DHL_FEEDBACK':
            shift_bit=10
        elif tag == 'DHL_NOISE_SCAN':
            shift_bit=24
        elif tag == 'DHL_USER_0':
            shift_bit=48
        elif tag == 'DHL_USER_1':
            shift_bit=49
        elif tag == 'DHL_USER_2':
            shift_bit=50            
        elif tag == 'DHL_USER_3':
            shift_bit=51            
        elif tag == 'DHL_USER_4':
            shift_bit=52        
            
        return np.bool_(np.int_(flag_bin[64-1-shift_bit]))
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    