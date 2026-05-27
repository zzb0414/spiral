# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 17:24:59 2021

@author: yongquan.ye@united-imaging.com
"""
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class UIHPPARecon:
    def __init__(self,prot):
        self.Prot = prot
        self.KernelLength = np.ones([2],dtype=int)*5
        self.HalfKernelLength =np.int32((self.KernelLength-1)/2)
        self.KernelBlockSize = np.prod(self.KernelLength)        
        self.RefIndDim = []
        self.RefLineInd=np.ndarray([0],dtype='int32')
        
        self.KernelPat  = np.zeros([3,self.KernelLength[0],self.KernelLength[1]], dtype='int32')                
        self.KernelPat[0] = [[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0],[1,1,1,1,1],[0,0,0,0,0]] # for 2x PPA 
        # add your own PPA pattern if necessary
        # self.KernelPat[1] = [[0,1,1,1,0],[0,0,0,0,0],[1,1,0,1,1],[0,0,0,0,0],[0,1,1,1,0]] # for UFI
        # self.KernelPat[2] = [[0,1,0,1,0],[0,1,0,1,0],[0,1,0,1,0],[0,1,0,1,0],[0,1,0,1,0]] # for UFI
        
    def GetRefBlockInd(self,kSpace):        
        dim = kSpace.shape          
        for ipe in range(self.HalfKernelLength[0],dim[0]-self.HalfKernelLength[0]): 
            for ispe in range(self.HalfKernelLength[1],dim[1]-self.HalfKernelLength[1]):
                if 0 != np.prod(kSpace[ipe-self.HalfKernelLength[0]:ipe+self.HalfKernelLength[0],ispe-self.HalfKernelLength[1]:ispe+self.HalfKernelLength[1]]):
                # if abs(ipe-dim[0]/2)<24 and abs(ispe-dim[1]/2)<24 :
                    self.RefLineInd = np.append(self.RefLineInd,[ipe,ispe])
        
        length = int(np.size(self.RefLineInd)/2)
        self.RefLineInd = np.reshape(self.RefLineInd,[length,2])
        # print (np.shape(self.RefLineInd))
        
    
    def CalcPPACoef(self,DataIn2D):
        self.RefIndDim = np.shape(self.RefLineInd) # Nx2, 2 being [PE,SPE]
        DatDim = np.shape(DataIn2D) # CHxPExSPE
        self.ChNum = DatDim[0]
        res = np.zeros([self.ChNum,self.KernelBlockSize,self.RefIndDim[0]],dtype='complex64')
        
        for ipe in range (-self.HalfKernelLength[0],self.HalfKernelLength[0]+1):
            for ispe in range (-self.HalfKernelLength[1],self.HalfKernelLength[1]+1):
                LineTmp = np.zeros([self.ChNum,self.RefIndDim[0]],dtype='complex64')
                for jj in range(self.RefIndDim[0]):        
                    LineTmp[:,jj] = DataIn2D[:, self.RefLineInd[jj,0] + ipe, self.RefLineInd[jj,1] + ispe]
                ind = (ipe+self.HalfKernelLength[0]) + (ispe+self.HalfKernelLength[1])*self.KernelLength[0]
                res[:,ind,:] = LineTmp
        
        res = np.reshape(res,[self.KernelBlockSize*self.ChNum,self.RefIndDim[0]])        
        CoefMat = np.matmul(res,np.conj(np.transpose(res)))     
        CoefMat = np.transpose(CoefMat)
            
        return CoefMat
    
    
    #extract PPA coefficients for the given block pattern
    def CalcPPAKernel(self,CoefMat,Pattern):        
        Aty1 = np.ndarray([np.size(Pattern)*self.ChNum, self.ChNum],dtype = 'complex64' )
        for ich in range(self.ChNum):
            IdxY = np.int32((self.KernelBlockSize-1)/2 + ich * self.KernelBlockSize)            
            Aty1[:,ich] = CoefMat[:,IdxY]
        
        Aty2 = np.ndarray([np.sum(Pattern)*self.ChNum, self.ChNum],dtype = 'complex64' )
        AtA1 = np.ndarray([np.sum(Pattern)*self.ChNum, np.size(Pattern)*self.ChNum],dtype = 'complex64' )
        cnt = -1
        for ich in range(self.ChNum):
            for ii in range(self.KernelBlockSize):
                if Pattern[ii%self.KernelLength[0]][int(ii/self.KernelLength[0])]==1:
                    IdxA = ii + ich*self.KernelBlockSize
                    cnt += 1
                    Aty2[cnt,:] = Aty1[IdxA,:]
                    AtA1[cnt,:] = CoefMat[IdxA,:]
        
        AtA2 = np.ndarray([np.sum(Pattern)*self.ChNum, np.sum(Pattern)*self.ChNum],dtype = 'complex64' )
        cnt = -1
        for ich in range(self.ChNum):
            for ii in range(self.KernelBlockSize):
                if Pattern[ii%self.KernelLength[0]][int(ii/self.KernelLength[0])]==1:
                    IdxA = ii + ich*self.KernelBlockSize       
                    cnt += 1
                    AtA2[:, cnt] = AtA1[:,IdxA]
                    
        eyeMat = np.eye(np.shape(AtA2)[0])        
        w = np.linalg.norm(AtA2,ord='fro')/np.shape(AtA2)[0]*0.01
        # print('Lambda is ' + str(w))
        RawKernel = np.matmul(np.linalg.inv(AtA2+eyeMat*w),Aty2)
        Kernel = np.zeros([self.KernelBlockSize*self.ChNum,self.ChNum],dtype='complex64')
                
        cnt = -1
        for ich in range(self.ChNum):
           for ii in range(self.KernelBlockSize):
               if Pattern[ii%self.KernelLength[0]][int(ii/self.KernelLength[0])]==1:
                   IdxA = ii + ich*self.KernelBlockSize
                   cnt += 1
                   Kernel[IdxA,:] = RawKernel[cnt,:]
       
        return Kernel
    
    # actually filling the missing lines
    def PPAFillMissingLines (self, DataIn2D, Pattern, Kernel):     
        DataOut2D = np.zeros(np.shape(DataIn2D),dtype='complex64')
        DatDim = np.shape(DataIn2D)  # CHxPExSPE
        for ipe in range(self.HalfKernelLength[0],DatDim[1]-self.HalfKernelLength[0]): 
            for ispe in range(self.HalfKernelLength[1],DatDim[2]-self.HalfKernelLength[1]):   
                if np.abs(DataIn2D[1, ipe, ispe]) ==0:
                    cBlock =DataIn2D[:, ipe-self.HalfKernelLength[0]:ipe+self.HalfKernelLength[0]+1,
                                     ispe-self.HalfKernelLength[1]:ispe+self.HalfKernelLength[1]+1]
                    cPat =  (0 != np.abs(cBlock[1,:,:]))*Pattern
                    if np.sum(cPat) == np.sum(Pattern):
                        cBlock = np.moveaxis(cBlock, 1, -1)
                        for ich in range (self.ChNum):
                            DataOut2D[ich,ipe,ispe] = np.sum(Kernel[:,ich] * np.ndarray.flatten(cBlock))    
        return DataOut2D
    
    # find PPA pattern, this can be very flexible
    def FindPattern(self, DataRef2D,DataUpdated2D, MinRatio):
        DatDim = np.shape(DataRef2D)  
        for ipe in range(self.HalfKernelLength[0],DatDim[1]-self.HalfKernelLength[0]): 
            for ispe in range(self.HalfKernelLength[1],DatDim[2]-self.HalfKernelLength[1]):  
                 if np.abs(DataUpdated2D[1, ipe, ispe]) == 0:
                     cBlock =DataRef2D[:, ipe-self.HalfKernelLength[0]:ipe+self.HalfKernelLength[0]+1,
                                     ispe-self.HalfKernelLength[1]:ispe+self.HalfKernelLength[1]+1]
                     Pattern = np.int32(0 != np.abs(cBlock[1,:,:]))
                     PatRatio = np.sum(Pattern)/np.size(Pattern)                     
                     if  PatRatio>= MinRatio:
                         return Pattern
        return []
        
    # Run Forest, run!
    def Run(self, DataIn,procnum,return_dict): 
        #DataIn is undersampled k-sapce data, ffted over RO direction
        #DataIn dimension: CHxROxPExSPE(SLC)            
        Dim = np.int(self.Prot.find('.//Dimension/Value').text)
        if Dim==1:
            ScanMode = '2D'
        else:
            ScanMode = '3D'                
       
        if ScanMode == '3D':
            dim = DataIn.shape    #CHxROxPExSPE
            
            # RO at this point is already iffted and cliped            
            dim = DataIn.shape
            DataOut = np.zeros(dim,dtype='complex64')
            
            self.GetRefBlockInd(DataIn[1,1,:,:])    
                    
            for iRO in range(dim[1]):                
                # print('Recon on iRO = ' + str(iRO+1))                
                DatIn2D = DataIn[:,iRO,:,:]                     
                
                #create CalibKernel: KernelLength*KernelLength*ch*ch
                CoefMat = self.CalcPPACoef(DatIn2D)
                
                DataOut2D = np.copy(DatIn2D)                
                # firstly, use predefined pattern for recon                        
                for ii in range(np.shape(self.KernelPat)[0]):
                    Kernel = self.CalcPPAKernel(CoefMat,self.KernelPat[ii])        
                    DataOut2D_newlines = self.PPAFillMissingLines (DataOut2D, self.KernelPat[ii], Kernel)
                    DataOut2D += DataOut2D_newlines
                 
                # automatically find more good patterns
                # usually turned off unless you have more undersampling patterns
                NewPatCnt = 0
                while False:
                    FoundPattern = self.FindPattern(DatIn2D, DataOut2D, 8/25)
                    if np.size(FoundPattern) == 0:
                        break
                    Kernel = self.CalcPPAKernel(CoefMat,FoundPattern)
                    DataOut2D_newlines = self.PPAFillMissingLines (DataOut2D, FoundPattern, Kernel)
                    DataOut2D += DataOut2D_newlines
                    NewPatCnt += 1
                    # LinCnt = np.sum(np.int32(0 != np.abs(DataOut2D[1,:,:])))
                    # print(LinCnt)
                    # print(FoundPattern)    
                if NewPatCnt > 0:
                    print('Found '+str(NewPatCnt)+' New pattern')
                
                #True:  output in image domain
                #False: output in k space domain
                if True: 
                    DataOut2D = np.fft.fftshift(DataOut2D, axes = [1,2])
                    DataOut2D = np.fft.ifftn(DataOut2D,axes=[1,2])                    
                    DataOut2D = np.fft.ifftshift(DataOut2D, axes = [1,2])
                DataOut[:,iRO,:,:] = DataOut2D
                 
        else: # 2D recon mode, input is single slice
            dim = DataIn.shape    #CHxROxPE      
            # transform RO back to k space
            DataIn = np.fft.fftshift(DataIn, axes = [1,])
            DataIn = np.fft.fft(DataIn,axis=1)    #CHxROxPE  
            
            DataIn = np.moveaxis(DataIn, 1, -1)  #CHxPExRO
            
            dim = DataIn.shape                
            DataOut = np.zeros(dim,dtype='complex64')
            
            self.GetRefBlockInd(DataIn[1,:,:])                
                                   
            CoefMat = self.CalcPPACoef(DataIn)
            
            DataOut2D = np.copy(DataIn)         

            # firstly, use predefined pattern for recon                        
            
            for ii in range(np.shape(self.KernelPat)[0]):
                Kernel = self.CalcPPAKernel(CoefMat,self.KernelPat[ii])        
                DataOut2D_newlines = self.PPAFillMissingLines (DataOut2D, self.KernelPat[ii], Kernel)
                DataOut2D += DataOut2D_newlines            
                                  
            # automatically find more good patterns
            # usually turned off unless you have more undersampling patterns
            NewPatCnt=0
            while False:
                FoundPattern = self.FindPattern(DataIn, DataOut2D, 5/25)
                if np.size(FoundPattern) == 0:
                    break
                Kernel = self.CalcPPAKernel(CoefMat,FoundPattern)
                DataOut2D_newlines = self.PPAFillMissingLines (DataOut2D, FoundPattern, Kernel)
                DataOut2D += DataOut2D_newlines
                NewPatCnt += 1             
            if NewPatCnt > 0:
                print('Found '+str(NewPatCnt)+' New pattern')
            
            #True:  output in image domain
            #False: output in k space domain
            if True: 
                DataOut2D = np.fft.fftshift(DataOut2D, axes = [1,2])
                DataOut2D = np.fft.ifftn(DataOut2D,axes=[1, 2])                
                DataOut2D = np.fft.ifftshift(DataOut2D, axes = [1,2])      
                
            DataOut2D = np.moveaxis(DataOut2D, 1, -1)                
            DataOut = DataOut2D            
        
        return_dict[procnum] = DataOut        
                 
        
        
        
        
        
        
        
        
        
   