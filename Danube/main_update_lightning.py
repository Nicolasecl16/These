#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:59:23 2020
@author: rfablet
"""
print('OK')
#######################################

import argparse
import numpy as np
import matplotlib.pyplot as plt 
#import os
#import tensorflow.keras as keras
import xarray as xr

import time
import copy
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import scipy
from scipy.integrate import solve_ivp
from sklearn.feature_extraction import image
from netCDF4 import Dataset
import datetime

# specific torch module 
#import dinAE_solver_torch as dinAE
import Updated_Solver as NN_4DVar
#import torch4DVarNN_solver as NN_4DVar
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from netCDF4 import Dataset
from models_Copy1 import  Gradient_img, LitModel

##PART 1 : DATA LOADING

############################

parser = argparse.ArgumentParser()
flagProcess    = [0,1,2,3,4]#Sequence fo processes to be run
    
flagRandomSeed = 0
flagSaveModel  = 1
     
batch_size  = 96#4#4#8#12#8#256#8

dirSAVE     = './ResDanube4DVar/'
suffix_exp='exp2'
genFilename = 'Debit_v11'
  
flagAEType = 2 # 0: L96 model, 1-2: GENN
DimAE      = 50#50#10#50
    
UsePriodicBoundary = True # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
InterpFlag         = False

NbDays          = 18244

time_step  = 1
DT = 21
sigNoise   = np.sqrt(2)
rateMissingData = 0.5#0.9
width_med_filt_spatial = 5
width_med_filt_temp = 1


# loss weghing wrt time
w_ = np.zeros(DT)
w_[int(DT / 2)] = 1.
wLoss = torch.Tensor(w_)

flagTypeMissData = 3

#####################################



#import torch.distributed as dist

## NN architectures and optimization parameters
#batch_size      = 2#16#4#4#8#12#8#256#
#DimAE           = 50#10#10#50
#dimGradSolver   = 100 # dimension of the hidden state of the LSTM cell
#rateDropout     = 0.25 # dropout rate 
#flag_aug_state = 2#True#
#flag_augment_training_data = True#False#

# data generation
#sigNoise = 0. ## additive noise standard deviation
#flagSWOTData = True #False # rue ## use SWOT data or not
#flagNoSSTObs = False #True #
#flag_vv  = 'vv_10m'

#width_med_filt_spatial = 5
#width_med_filt_temp = 1

#dT              = 5 ## Time window of each space-time patch
#W               = 200 ## width/height of each space-time patch
#dx              = 1 ## subsampling step if > 1
#Nbpatches       = 1#10#10#25 ## number of patches extracted from each time-step 
#rnd1            = 0 ## random seed for patch extraction (space sam)
#rnd2            = 100 ## random seed for patch extraction
#dwscale         = 1

# loss
#p_norm_loss = 2. 
#q_norm_loss = 2. 
#r_norm_loss = 2. 
#thr_norm_loss = 0.

#W = int(W/dx)

#UsePriodicBoundary = False # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
#InterpFlag         = False # True => force reconstructed field to observed data after each gradient-based update

print('........ Data extraction')
if flagRandomSeed == 0:
    print('........ Random seed set to 100')
    np.random.seed(100)
        
###############################################################
## data extraction
ncfile = Dataset('Dataset_danube.nc',"r")
L=[]
for i in range(31):
    L.append(ncfile['S'+str(i+1)][:].reshape(18244,1))
        
dataset = np.concatenate((L[0],L[1],L[2],L[3],L[4],L[5],L[6],L[7],L[8],L[9],L[10],L[11],L[12],L[13],L[14],L[15],L[16],L[17],L[18],L[19],L[20],L[21],L[22],L[23],L[24],L[25],L[26],L[27],L[28],L[29],L[30]),axis=1)

# Definiton of training, validation and test dataset    
i=0
Indtrain=[]
Indval=[]
Indtest=[]
while (i+1)*395<(NbDays-1):
    x=395*i
    Indtrain.append([x,(x+305)])
    Indval.append([(x+319),(x+350)])
    Indtest.append([x+364,x+395])
    i+=1
    

#Se restreindre à l'été car pas de pluie??
day0=datetime.date(1960,1,1)
dayend=datetime.date(2009,12,12)


#Trouver une valeur de seuil pour étudier le coût KL au-dessus du seuil : on se place dans flagTypeProcess 3 avec la station 4 masquée

D=dataset[Indtrain[0][0]:Indtrain[0][1],3]
for k in Indtrain[1::]:
    D=np.concatenate((D,dataset[k[0]:k[1],3]),axis=0)
r=0.1 #fraction supérieure 
D.sort()
seuil=D[int((1-r)*len(D))]
    
    ####################################################
## Generation of training  validationand test dataset
## Extraction of time series of dT time steps
#NbTraining = 6000#2000
#NbTest     = 256#256#500
#NbVal = ?
    
dataTrainingNoNaND = image.extract_patches_2d(dataset[Indtrain[0][0]:Indtrain[0][1],:],(DT,31)) 
for k in Indtrain[1::]:
    d= image.extract_patches_2d(dataset[k[0]:k[1],:],(DT,31))
    dataTrainingNoNaND=np.concatenate((dataTrainingNoNaND,d),axis=0)
        
    
dataValNoNaND = image.extract_patches_2d(dataset[Indval[0][0]:Indval[0][1],:],(DT,31))    
for k in Indval[1::]:
    d= image.extract_patches_2d(dataset[k[0]:k[1],:],(DT,31))
    dataValNoNaND=np.concatenate((dataValNoNaND,d),axis=0)
print(dataValNoNaND.shape )  
    
dataTestNoNaND = image.extract_patches_2d(dataset[Indtest[0][0]:Indtest[0][1],:],(DT,31))
for k in Indtest[1::]:
    d= image.extract_patches_2d(dataset[k[0]:k[1],:],(DT,31))
    dataTestNoNaND=np.concatenate((dataTestNoNaND,d),axis=0)
print(dataTestNoNaND.shape ) 
        
# create missing data
#flagTypeMissData = 0 : Missing data randomly chosen on the patch driven by rateMissingData
#flagTypeMissData = 1 : Almost the same
#flagTypeMissData = 2 : In each patch, different station are randomly chosen and are masked according to rateMissingData
#flagTypeMissData = 3 : The same stations listed in MaskedStations are masked
flagTypeMissData = 2
if flagTypeMissData == 0:
    indRandD         = np.random.permutation(dataTrainingNoNaND.shape[0]*dataTrainingNoNaND.shape[1]*dataTrainingNoNaND.shape[2])
    indRandD         = indRandD[0:int(rateMissingData*len(indRandD))]
    dataTrainingD    = np.copy(dataTrainingNoNaND).reshape((dataTrainingNoNaND.shape[0]*dataTrainingNoNaND.shape[1]*dataTrainingNoNaND.shape[2],1))
    dataTrainingD[indRandD] = float('nan')
    dataTrainingD    = np.reshape(dataTrainingD,(dataTrainingNoNaND.shape[0],dataTrainingNoNaND.shape[1],dataTrainingNoNaND.shape[2]))
            
    indRandD         = np.random.permutation(dataValNoNaND.shape[0]*dataValNoNaND.shape[1]*dataValNoNaND.shape[2])
    indRandD         = indRandD[0:int(rateMissingData*len(indRandD))]
    dataValD    = np.copy(dataValNoNaND).reshape((dataValNoNaND.shape[0]*dataValNoNaND.shape[1]*dataValNoNaND.shape[2],1))
    dataValD[indRandD] = float('nan')
    dataValD    = np.reshape(dataValD,(dataValNoNaND.shape[0],dataValNoNaND.shape[1],dataValNoNaND.shape[2]))
            
            
    indRandD         = np.random.permutation(dataTestNoNaND.shape[0]*dataTestNoNaND.shape[1]*dataTestNoNaND.shape[2])
    indRandD         = indRandD[0:int(rateMissingData*len(indRandD))]
    dataTestD        = np.copy(dataTestNoNaND).reshape((dataTestNoNaND.shape[0]*dataTestNoNaND.shape[1]*dataTestNoNaND.shape[2],1))
    dataTestD[indRandD] = float('nan')
    dataTestD          = np.reshape(dataTestD,(dataTestNoNaND.shape[0],dataTestNoNaND.shape[1],dataTestNoNaND.shape[2]))

    genSuffixObs    = '_ObsRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
        
elif flagTypeMissData==1:
    time_step_obs   = int(1./(1.-rateMissingData))
    dataTrainingD    = np.zeros((dataTrainingNoNaND.shape))
    dataTrainingD[:] = float('nan')
            
    dataValD    = np.zeros((dataValNoNaND.shape))
    dataValD[:] = float('nan')
            
    dataTestD        = np.zeros((dataTestNoNaND.shape))
    dataTestD[:]     = float('nan')
               
    if 1*0:
                
        dataTrainingD[:,::time_step_obs,:] = dataTrainingNoNaND[:,::time_step_obs,:]
        dataValD[:,::time_step_obs,:] = dataValNoNaND[:,::time_step_obs,:]
        dataTestD[:,::time_step_obs,:]     = dataTestNoNaND[:,::time_step_obs,:]
                    
        genSuffixObs    = '_ObsSub_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
    else:
        for nn in range(0,dataTrainingD.shape[1],time_step_obs):
            indrand = np.random.permutation(dataTrainingD.shape[2])[0:int(0.5*dataTrainingD.shape[1])]
            dataTrainingD[:,nn,indrand] = dataTrainingNoNaND[:,nn,indrand]
                    
        for nn in range(0,dataTrainingD.shape[1],time_step_obs):
            indrand = np.random.permutation(dataTrainingD.shape[2])[0:int(0.5*dataTrainingD.shape[1])]
            dataValD[:,nn,indrand] = dataValNoNaND[:,nn,indrand]
                    
        for nn in range(0,dataTrainingD.shape[1],time_step_obs):
            indrand = np.random.permutation(dataTrainingD.shape[2])[0:int(0.5*dataTrainingD.shape[1])]
            dataTestD[:,nn,indrand] = dataTestNoNaND[:,nn,indrand]

        genSuffixObs    = '_ObsSubRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
        
elif flagTypeMissData == 2 :
    #
    Nbtraining=13110
    Nbval=506
    Nbtest=506
            
    ratemissingdata_space = 0.15
    time_step_obs   = int(1./(1.-rateMissingData))
    dataTrainingD    = np.zeros(([Nbtraining,dataTrainingNoNaND.shape[1],dataTrainingNoNaND.shape[2]]))
    dataTrainingD[:] = float('nan')
    dataTrainingNoNaND2    = np.zeros(([Nbtraining,dataTrainingNoNaND.shape[1],dataTrainingNoNaND.shape[2]]))
    dataTrainingNoNaND2[:] = float('nan')
            
    dataValD    = np.zeros(([Nbval,dataTrainingNoNaND.shape[1],dataTrainingNoNaND.shape[2]]))           
    dataValD[:] = float('nan')
    dataValNoNaND2=np.zeros(([Nbval,dataTrainingNoNaND.shape[1],dataTrainingNoNaND.shape[2]]))
    dataValNoNaND2[:] = float('nan')
            
    dataTestD        = np.zeros(([Nbtest,dataTrainingNoNaND.shape[1],dataTrainingNoNaND.shape[2]]))
    dataTestD[:]     = float('nan') 
    dataTestNoNaND2 =np.zeros(([Nbtest,dataTrainingNoNaND.shape[1],dataTrainingNoNaND.shape[2]]))
    dataTestNoNaND2[:] = float('nan')
            
    ind=0
    print(dataTrainingD.shape)
    
    while ind<Nbtraining:
        indrand=np.random.permutation(dataTrainingD.shape[2])[0:int((1-ratemissingdata_space)*dataTrainingD.shape[2])]
        dataTrainingD[ind,:,indrand]=dataTrainingNoNaND[ind%dataTrainingNoNaND.shape[0],:,indrand]
        dataTrainingNoNaND2[ind,:,:]=dataTrainingNoNaND[ind%dataTrainingNoNaND.shape[0],:,:]
                
        if ind <Nbval:
            indrand2=np.random.permutation(dataTrainingD.shape[2])[0:int((1-ratemissingdata_space)*dataTrainingD.shape[2])]
            dataValD[ind,:,indrand2]=dataValNoNaND[ind%dataValNoNaND.shape[0],:,indrand2]
            dataValNoNaND2[ind,:,:]=dataValNoNaND[ind%dataValNoNaND.shape[0],:,:]
                
            indrand3=np.random.permutation(dataTrainingD.shape[2])[0:int((1-ratemissingdata_space)*dataTrainingD.shape[2])]
            dataTestD[ind,:,indrand3]=dataTestNoNaND[ind%dataTestNoNaND.shape[0],:,indrand3]
            dataTestNoNaND2[ind,:,:] = dataTestNoNaND[ind%dataTestNoNaND.shape[0],:,:]
        ind+=1
                
    dataTrainingNoNaND =dataTrainingNoNaND2
    dataValNoNaND = dataValNoNaND2
    dataTestNoNaND =dataTestNoNaND2        
    genSuffixObs    = '_ObsSubRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2) 
        
#mask only on specific station
else :
    MaskedStations=[2,4,16,25]
            
    dataTrainingD    = np.zeros((dataTrainingNoNaND.shape))
    dataTrainingD[:] = float('nan')
            
    dataValD    = np.zeros((dataValNoNaND.shape))
    dataValD[:] = float('nan')
            
    dataTestD        = np.zeros((dataTestNoNaND.shape))
    dataTestD[:]     = float('nan')
    print(dataTrainingNoNaND[0,:,:])  
    for i in range(31):
        dataTrainingD[:,:,i] = dataTrainingNoNaND[:,:,i]
        dataValD[:,:,i] = dataValNoNaND[:,:,i]
        dataTestD[:,:,i] = dataTestNoNaND[:,:,i]
    for i in MaskedStations:
        dataTrainingD[:,:,i-1] = float('nan')
        dataValD[:,:,i-1] = float('nan')
        dataTestD[:,:,i-1] = float('nan')
    genSuffixObs    = '_ObsSubRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
    print(dataTrainingNoNaND[0,:,:])    
print('... Data type: '+genSuffixObs)
    #for nn in range(0,dataTraining.shape[1],time_step_obs):
    #    dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
    #dataTest    = np.zeros((dataTestNoNaN.shape))
    #dataTest[:] = float('nan')
    #dataTest[:,::time_step_obs,:] = dataTestNoNaN[:,::time_step_obs,:]
        
# set to NaN patch boundaries    
if 1*0:
    dataTrainingD[:,0:10,:] =  float('nan')
    dataValD[:,0:10,:] =  float('nan')
    dataTestD[:,0:10,:]     =  float('nan')
    dataTrainingD[:,dT-10:dT,:] =  float('nan')
    dataValD[:,dT-10:dT,:] =  float('nan')
    dataTestD[:,dT-10:dT,:]     =  float('nan')
            
                                
# mask for NaN
maskTrainingD = (dataTrainingD == dataTrainingD).astype('float')
maskValD = (dataValD == dataValD).astype('float')
maskTestD     = ( dataTestD    ==  dataTestD   ).astype('float')
            
dataTrainingD = np.nan_to_num(dataTrainingD)
        
dataValD = np.nan_to_num(dataValD)
dataTestD     = np.nan_to_num(dataTestD)
            
    # Permutation to have channel as #1 component
dataTrainingD      = np.moveaxis(dataTrainingD,-1,1)
maskTrainingD      = np.moveaxis(maskTrainingD,-1,1)
dataTrainingNoNaND = np.moveaxis(dataTrainingNoNaND,-1,1)
        
dataValD      = np.moveaxis(dataValD,-1,1)
maskValD      = np.moveaxis(maskValD,-1,1)
dataValNoNaND = np.moveaxis(dataValNoNaND,-1,1)
            
dataTestD      = np.moveaxis(dataTestD,-1,1)
maskTestD      = np.moveaxis(maskTestD,-1,1)
dataTestNoNaND = np.moveaxis(dataTestNoNaND,-1,1)
            
# set to NaN patch boundaries
#dataTraining[:,0:5,:] =  dataTrainingNoNaN[:,0:5,:]
#dataTest[:,0:5,:]     =  dataTestNoNaN[:,0:5,:]
    
############################################
## raw data
X_trainD         = dataTrainingNoNaND
        
X_train_missingD = dataTrainingD
mask_trainD      = maskTrainingD
        
X_valD         = dataValNoNaND
X_val_missingD = dataValD
mask_valD      = maskValD
        
X_testD         = dataTestNoNaND
X_test_missingD = dataTestD
mask_testD      = maskTestD
            
############################################
## normalized data wrt to each measurement station
        
if flagTypeMissData ==2 :
    mean2 = np.mean(X_train_missingD[:],0)
    mean2mask = np.mean(mask_trainD[:],0)


    mean3 = np.mean(mean2,1)
    mean3 = mean3.reshape(31,1)
    print(mean3)
    mean3mask = np.mean(mean2mask,1)
    mean3mask = mean3mask.reshape(31,1)
    print(mean3mask)
    meanTr          = mean3/mean3mask
    print(meanTr)
    mean2true = np.mean(X_trainD[:],0)
    mean3true = np.mean(mean2true,1)
    mean3true = mean3true.reshape(31,1)
    meanTrtrue = mean3true
    print(meanTrtrue)
            
    meansquaretrue = np.mean( (X_trainD-meanTrtrue)**2,0)
    meansquare2true = np.mean(meansquaretrue,1)
    meansquare2true=meansquare2true.reshape(31,1)
    stdTrtrue           = np.sqrt(meansquare2true )
    print(stdTrtrue)
            
            
    x_train_missingD = X_train_missingD - meanTr*mask_trainD
            
    x_val_missingD = X_val_missingD - meanTr*mask_valD
    x_test_missingD  = X_test_missingD - meanTr*mask_testD
            
    # scale wrt to each station
    meansquare = np.mean( X_train_missingD**2,0)
    meansquare2 = np.mean(meansquare,1)
    meansquare2=meansquare2.reshape(31,1)
    stdTr           = np.sqrt(meansquare2 / mean3mask)
            
    x_train_missingD = x_train_missingD / stdTr
    x_val_missingD = x_val_missingD / stdTr
    x_test_missingD  = x_test_missingD / stdTr
            
    x_trainD = (X_trainD - meanTr) / stdTr
    x_valD = (X_valD - meanTr) / stdTr
    x_testD  = (X_testD - meanTr) / stdTr
            
    print(np.mean(x_train_missingD))
    print(np.mean(x_trainD))
    print(np.mean(x_val_missingD))
    print(np.mean(x_valD))
    print(np.mean(x_test_missingD))
    print(np.mean(x_testD))
            
            
               
elif flagTypeMissData==3 :
    mean2 = np.mean(X_train_missingD[:],0)
    mean3 = np.mean(mean2,1)
    mean3 = mean3.reshape(31,1)
    meanTr = mean3
    x_train_missingD = X_train_missingD - meanTr
    x_val_missingD = X_val_missingD - meanTr
    x_test_missingD  = X_test_missingD - meanTr
    meansquare = np.mean( x_train_missingD**2,0)
    meansquare2 = np.mean(meansquare,1)
    meansquare2=meansquare2.reshape(31,1)
            
    stdTr           = np.sqrt(meansquare2)
            
    for i in MaskedStations :
        stdTr[i-1] =1
    print(stdTr)
    print(X_trainD[0,:,:])
    x_train_missingD = x_train_missingD / stdTr
    x_val_missingD = x_val_missingD / stdTr
    x_test_missingD  = x_test_missingD / stdTr
    mean2true = np.mean(X_trainD[:],0)
    mean3true = np.mean(mean2true,1)
    mean3true = mean3true.reshape(31,1)
    meanTrtrue = mean3true
            
    meansquaretrue = np.mean( (X_trainD-meanTrtrue)**2,0)
    meansquare2true = np.mean(meansquaretrue,1)
    meansquare2true=meansquare2true.reshape(31,1)
    stdTrtrue           = np.sqrt(meansquare2true )
    print(stdTrtrue)
    print(meanTrtrue)
            
    x_trainD = (X_trainD - meanTrtrue) / stdTrtrue
    x_valD = (X_valD - meanTrtrue) / stdTrtrue
    x_testD  = (X_testD - meanTrtrue) / stdTrtrue
            
            
            
else : 
    mean2 = np.mean(X_train_missingD[:],0)
    mean2mask = np.mean(mask_trainD[:],0)
            

    mean3 = np.mean(mean2,1)
    mean3 = mean3.reshape(31,1)
    print(mean3)
    mean3mask = np.mean(mean2mask,1)
    mean3mask = mean3mask.reshape(31,1)
    print(mean3mask)
    meanTr          = mean3/mean3mask
    print(meanTr)
            
    x_train_missingD = X_train_missingD - meanTr*mask_trainD
    x_val_missingD = X_val_missingD - meanTr
    x_test_missingD  = X_test_missingD - meanTr
            
    # scale wrt to each station
    meansquare = np.mean( X_train_missingD**2,0)
    meansquare2 = np.mean(meansquare,1)
    meansquare2=meansquare2.reshape(31,1)
    stdTr           = np.sqrt(meansquare2 / mean3mask)
            
    x_train_missingD = x_train_missingD / stdTr
    x_val_missingD = x_val_missingD / stdTr
    x_test_missingD  = x_test_missingD / stdTr
            
    x_trainD = (X_trainD - meanTr) / stdTr
    x_valD = (X_valD - meanTr) / stdTr
    x_testD  = (X_testD - meanTr) / stdTr
            
# Generate noisy observsation
        
#X_train_obsD = X_train_missingD + sigNoise * maskTrainingD * np.random.randn(X_train_missingD.shape[0],X_train_missingD.shape[1],X_train_missingD.shape[2])
#X_val_obsD = X_val_missingD + sigNoise * maskValD * np.random.randn(X_val_missingD.shape[0],X_val_missingD.shape[1],X_val_missingD.shape[2])
#X_test_obsD  = X_test_missingD  + sigNoise * maskTestD * np.random.randn(X_test_missingD.shape[0],X_test_missingD.shape[1],X_test_missingD.shape[2])
            
#x_train_obsD = (X_train_obsD - meanTr) / stdTr
#x_val_obsD = (X_val_obsD - meanTr) / stdTr
#x_test_obsD  = (X_test_obsD - meanTr) / stdTr
        
#Without noise :
X_train_obsD = X_train_missingD 
X_val_obsD = X_val_missingD 
X_test_obsD  = X_test_missingD
        
x_train_obsD = x_train_missingD
x_val_obsD = x_val_missingD
x_test_obsD = x_test_missingD 
        
print('..... Training dataset: %dx%dx%d'%(x_train_missingD.shape[0],x_trainD.shape[1],x_trainD.shape[2]))
print('..... Validation dataset: %dx%dx%d'%(x_valD.shape[0],x_valD.shape[1],x_valD.shape[2]))
print('..... Test dataset    : %dx%dx%d'%(x_testD.shape[0],x_testD.shape[1],x_testD.shape[2]))
            

print('........ Initialize interpolated states')
## Initial interpolation
flagInit = 0
            
if flagInit == 0: 
    X_train_InitD = mask_trainD * X_train_obsD + (1. - mask_trainD) * (np.zeros(X_train_missingD.shape) + meanTr)
    X_val_InitD = mask_valD * X_val_obsD + (1. - mask_valD) * (np.zeros(X_val_missingD.shape) + meanTr)
    X_test_InitD  = mask_testD * X_test_obsD + (1. - mask_testD) * (np.zeros(X_test_missingD.shape) + meanTr)
else:
    X_train_InitD = np.zeros(X_trainD.shape)
    for ii in range(0,X_trainD.shape[0]):
        # Initial linear interpolation for each component
        XInitD = np.zeros((X_trainD.shape[1],X_trainD.shape[2]))
           
        for kk in range(0,mask_trainD.shape[1]):
            indt  = np.where( mask_trainD[ii,kk,:] == 1.0 )[0]
            indt_ = np.where( mask_trainD[ii,kk,:] == 0.0 )[0]
           
            if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_train_obsD[ii,kk,indt])
                XInitD[kk,indt]  = X_train_obsD[ii,kk,indt]
                XInitD[kk,indt_] = fkk(indt_)
            else:
                XInitD = XInitD + meanTr
            
        X_train_InitD[ii,:,:] = XInitD
            
    X_val_InitD = np.zeros(X_valD.shape)
    for ii in range(0,X_valD.shape[0]):
        # Initial linear interpolation for each component
        XInitD = np.zeros((X_valD.shape[1],X_valD.shape[2]))
           
        for kk in range(0,mask_valD.shape[1]):
            indt  = np.where( mask_valD[ii,kk,:] == 1.0 )[0]
            indt_ = np.where( mask_valD[ii,kk,:] == 0.0 )[0]
           
            if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_val_obsD[ii,kk,indt])
                XInitD[kk,indt]  = X_val_obsD[ii,kk,indt]
                XInitD[kk,indt_] = fkk(indt_)
            else:
                XInitD = XInitD + meanTr
            
        X_val_InitD[ii,:,:] = XInitD
            
    X_test_InitD = np.zeros(X_testD.shape)
    for ii in range(0,X_testD.shape[0]):
        # Initial linear interpolation for each component
        XInit = np.zeros((X_testD.shape[1],X_testD.shape[2]))
            
        for kk in range(0,X_testD.shape[1]):
            indt  = np.where( mask_testD[ii,kk,:] == 1.0 )[0]
            indt_ = np.where( mask_testD[ii,kk,:] == 0.0 )[0]
            
            if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_test_obsD[ii,kk,indt])
                XInit[kk,indt]  = X_test_obsD[ii,kk,indt]
                XInit[kk,indt_] = fkk(indt_)
            else:
                XInit = XInit + meanTr
        
        X_test_InitD[ii,:,:] = XInit
        #plt.figure()
        #plt.figure()
        #plt.plot(YObs[0:200,1],'r.')
        #plt.plot(XGT[0:200,1],'b-')
        #plt.plot(XInit[0:200,1],'k-')
                        
x_train_InitD = ( X_train_InitD - meanTr ) / stdTr
x_val_InitD = ( X_val_InitD - meanTr ) / stdTr
x_test_InitD = ( X_test_InitD - meanTr ) / stdTr
        
# reshape to dT-1 for time dimension
DT = DT-1
X_train_obsD        = X_train_obsD[:,:,0:DT]
X_trainD            = X_trainD[:,:,0:DT]
X_train_missingD    = X_train_missingD[:,:,0:DT]
mask_trainD         = mask_trainD[:,:,0:DT]
            
x_train_obsD        = x_train_obsD[:,:,0:DT]
x_trainD            = x_trainD[:,:,0:DT]
x_train_InitD       = x_train_InitD[:,:,0:DT]
X_train_InitD       = X_train_InitD[:,:,0:DT]
        
X_val_obsD        = X_val_obsD[:,:,0:DT]
X_valD            = X_valD[:,:,0:DT]
X_val_missingD    = X_val_missingD[:,:,0:DT]
mask_valD         = mask_valD[:,:,0:DT]
            
x_val_obsD        = x_val_obsD[:,:,0:DT]
x_valD            = x_valD[:,:,0:DT]
x_val_InitD       = x_val_InitD[:,:,0:DT]
X_val_InitD       = X_val_InitD[:,:,0:DT]

X_test_obsD        = X_test_obsD[:,:,0:DT]
X_testD            = X_testD[:,:,0:DT]
X_test_missingD    = X_test_missingD[:,:,0:DT]
mask_testD         = mask_testD[:,:,0:DT]

x_test_obsD        = x_test_obsD[:,:,0:DT]
x_testD            = x_testD[:,:,0:DT]
x_test_InitD       = x_test_InitD[:,:,0:DT]
X_test_InitD       = X_test_InitD[:,:,0:DT]

print('..... Training dataset: %dx%dx%d'%(x_trainD.shape[0],x_trainD.shape[1],x_trainD.shape[2]))
print('..... Validation dataset: %dx%dx%d'%(x_valD.shape[0],x_valD.shape[1],x_valD.shape[2]))
print('..... Test dataset    : %dx%dx%d'%(x_testD.shape[0],x_testD.shape[1],x_testD.shape[2]))
    
    
training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_InitD),torch.Tensor(x_train_obsD),torch.Tensor(mask_trainD),torch.Tensor(x_trainD)) # create your datset

val_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_val_InitD),torch.Tensor(x_val_obsD),torch.Tensor(mask_valD),torch.Tensor(x_valD)) # create your datset
     
test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_InitD),torch.Tensor(x_test_obsD),torch.Tensor(mask_testD),torch.Tensor(x_testD)) # create your datset

        
dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }            
dataset_sizes = {'train': len(training_dataset),'val': len(val_dataset), 'test': len(test_dataset)}
        
var_Tr    = np.var( x_trainD )
var_Tt    = np.var( x_testD )
var_Val   = np.var( x_valD )    

'''
###############################################################
        ## AE architecture        
print('........ Define AE architecture')
            
shapeData = np.ones(3).astype(int)
shapeData[1:] =  x_trainD.shape[1:]
                   
if flagAEType == 1: ## Conv model with no use of the central point
    dW = 2
    genSuffixModel = '_GENN_%d_%02d_%02d'%(flagAEType,DimAE,dW)
    class Encoder(torch.nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.conv1  = NN_4DVar.ConstrainedConv2d(1,DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)                      
            self.conv21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv3  = torch.nn.Conv2d(2*DimAE,1,(1,1),padding=0,bias=False)
            #self.conv4 = torch.nn.Conv1d(4*shapeData[0]*DimAE,8*shapeData[0]*DimAE,1,padding=0,bias=False)
            
            #self.conv2Tr = torch.nn.ConvTranspose1d(4*shapeData[0]*DimAE,8*shapeData[0]*DimAE,4,stride=4,bias=False)          
            #self.conv5 = torch.nn.Conv1d(8*shapeData[0]*DimAE,16*shapeData[0]*DimAE,3,padding=1,bias=False)
            #self.conv6 = torch.nn.Conv1d(16*shapeData[0]*DimAE,shapeData[0],3,padding=1,bias=False)
            
        def forward(self, xin):
            x_1 = torch.cat((xin[:,:,xin.size(2)-dW:,:],xin,xin[:,:,0:dW,:]),dim=2)
            #x_1 = x_1.view(-1,1,xin.size(1)+2*dW,xin.size(2))
            x   = self.conv1( x_1 )
            x   = x[:,:,dW:xin.size(2)+dW,:]
            x   = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)),dim=1)
            x   = self.conv3( x )
            #x = self.conv4( F.relu(x) )
            x = x.view(-1,shapeData[0],shapeData[1],shapeData[2])
            return x
    class Decoder(torch.nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            
        def forward(self, x):
            return torch.mul(1.,x)
                
elif flagAEType == 2: ## Conv model with no use of the central point
    dW = 5
    genSuffixModel = '_GENN_%d_%02d_%02d'%(flagAEType,DimAE,dW)
    class Encoder(torch.nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.pool1  = torch.nn.AvgPool2d((1,4))
            self.conv11 = NN_4DVar.ConstrainedConv2d(1,DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)                      
            self.conv12 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv3  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
            
            self.convTr = torch.nn.ConvTranspose2d(DimAE,DimAE,(1,4),stride=(1,4),bias=False)          
            #self.conv5 = torch.nn.Conv1d(8*shapeData[0]*DimAE,16*shapeData[0]*DimAE,3,padding=1,bias=False)
            #self.conv6 = torch.nn.Conv1d(16*shapeData[0]*DimAE,shapeData[0],3,padding=1,bias=False)
            self.conv11_1 = NN_4DVar.ConstrainedConv2d(1,DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)                      
            self.conv12_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv21_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv22_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv23_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv3_1  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
            
            self.convF    = torch.nn.Conv2d(DimAE,1,1,padding=0,bias=False)
        def forward(self, xin):
                
            x_1 = self.pool1(xin)
                
            x_1 = torch.cat((x_1[:,:,x_1.size(2)-dW:,:],x_1,x_1[:,:,0:dW,:]),dim=2)
                
            #x_1 = x_1.view(-1,1,xin.size(1)+2*dW,xin.size(2))
            x   = self.conv11( x_1 )
               
            x   = self.conv12( F.relu(x) )
                
            x   = x[:,:,dW:xin.size(2)+dW,:]
                
            x   = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)),dim=1)
                
            x   = self.conv3( x )
          
            x   = self.convTr( x )
                
                      
            x_2 = torch.cat((xin[:,:,xin.size(2)-dW:,:],xin,xin[:,:,0:dW,:]),dim=2)
                
            dx  = self.conv11_1( x_2 )
                
            dx  = self.conv12_1( F.relu(dx) )
                
            dx   = dx[:,:,dW:xin.size(2)+dW,:]
                
            dx   = torch.cat((self.conv21_1(dx), self.conv22_1(dx) * self.conv23_1(dx)),dim=1)
                
            dx   = self.conv3_1( dx )
                
                
            y=x+dx
            x    = self.convF( x + dx )
                
            #x = self.conv4( F.relu(x) )
            x = x.view(-1,shapeData[0],shapeData[1],shapeData[2])
                
            return x
    class Decoder(torch.nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            
        def forward(self, x):
            return torch.mul(1.,x)
            
elif flagAEType == 3: ## Same as flagAEType == 2 with no constraint on central point
    dW = 5
    genSuffixModel = '_GENN_%d_%02d_%02d'%(flagAEType,DimAE,dW)
    class Encoder(torch.nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.pool1  = torch.nn.AvgPool2d((1,4))
            self.conv11 = torch.nn.Conv2d(1,DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)                      
            self.conv12 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv3  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
            
            self.convTr = torch.nn.ConvTranspose2d(DimAE,DimAE,(1,4),stride=(1,4),bias=False)          
            #self.conv5 = torch.nn.Conv1d(8*shapeData[0]*DimAE,16*shapeData[0]*DimAE,3,padding=1,bias=False)
            #self.conv6 = torch.nn.Conv1d(16*shapeData[0]*DimAE,shapeData[0],3,padding=1,bias=False)
            self.conv11_1 = torch.nn.Conv2d(1,DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)                      
            self.conv12_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv21_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv22_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv23_1 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv3_1  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
            
            self.convF    = torch.nn.Conv2d(DimAE,1,1,padding=0,bias=False)
        def forward(self, xin):
            x_1 = self.pool1(xin)
            x_1 = torch.cat((x_1[:,:,x_1.size(2)-dW:,:],x_1,x_1[:,:,0:dW,:]),dim=2)
            #x_1 = x_1.view(-1,1,xin.size(1)+2*dW,xin.size(2))
            x   = self.conv11( x_1 )
            x   = self.conv12( F.relu(x) )
            x   = x[:,:,dW:xin.size(2)+dW,:]
            x   = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)),dim=1)
            x   = self.conv3( x )
            x   = self.convTr( x )
                      
            x_2 = torch.cat((xin[:,:,xin.size(2)-dW:,:],xin,xin[:,:,0:dW,:]),dim=2)
            print(xin.shape)
            print(xin.size(2))
            dx  = self.conv11_1( x_2 )
            dx  = self.conv12_1( F.relu(dx) )
            dx   = dx[:,:,dW:xin.size(2)+dW,:]
            dx   = torch.cat((self.conv21_1(dx), self.conv22_1(dx) * self.conv23_1(dx)),dim=1)
            dx   = self.conv3_1( dx )
                      
            x    = self.convF( x + dx )
            #x = self.conv4( F.relu(x) )
            x = x.view(-1,shapeData[0],shapeData[1],shapeData[2])
            return x
    class Decoder(torch.nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            
        def forward(self, x):
            return torch.mul(1.,x)
            
elif flagAEType == 4: ## Conv model with no use of the central point
    dW = 5
    genSuffixModel = '_GENN_%d_%02d_%02d'%(flagAEType,DimAE,dW)
    class Encoder(torch.nn.Module):
        def __init__(self):
            super(Encoder, self).__init__()
            self.pool1  = torch.nn.AvgPool2d(4)
            self.conv1  = NN_4DVar.ConstrainedConv2d(1,2*DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
            self.conv2  = torch.nn.Conv1d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
                      
            self.conv21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.conv3  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
            #self.conv4 = torch.nn.Conv1d(4*shapeData[0]*DimAE,8*shapeData[0]*DimAE,1,padding=0,bias=False)
            
            self.conv2Tr = torch.nn.ConvTranspose2d(DimAE,1,(4,4),stride=(4,4),bias=False)          
            #self.conv5 = torch.nn.Conv1d(2*shapeData[0]*DimAE,2*shapeData[0]*DimAE,3,padding=1,bias=False)
            #self.conv6 = torch.nn.Conv1d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)
            #self.conv6 = torch.nn.Conv1d(16*shapeData[0]*DimAE,shapeData[0],3,padding=1,bias=False)
            
            self.convHR1  = NN_4DVar.ConstrainedConv2d(1,2*DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
            self.convHR2  = torch.nn.Conv2d(2*DimAE,DimAE,(1,1),padding=0,bias=False)
                      
            self.convHR21 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.convHR22 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.convHR23 = torch.nn.Conv2d(DimAE,DimAE,(1,1),padding=0,bias=False)
            self.convHR3  = torch.nn.Conv2d(2*DimAE,1,(1,1),padding=0,bias=False)

        def forward(self, xinp):
            #x = self.fc1( torch.nn.Flatten(x) )
            #x = self.pool1( xinp )
            x = self.pool1( xinp )
            x = self.conv1( x )
            x = self.conv2( F.relu(x) )
            x = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)),dim=1)
            x = self.conv3( x )
            x = self.conv2Tr( x )
            #x = self.conv5( F.relu(x) )
            #x = self.conv6( F.relu(x) )
                      
            xHR = self.convHR1( xinp )
            xHR = self.convHR2( F.relu(xHR) )
            xHR = torch.cat((self.convHR21(xHR), self.convHR22(xHR) * self.convHR23(xHR)),dim=1)
            xHR = self.convHR3( xHR )
                      
            x   = torch.add(x,1.,xHR)
                      
            #x = x.view(-1,shapeData[0],shapeData[1],shapeData[2])
            return x
            
    class Decoder(torch.nn.Module):
        def __init__(self):
            super(Decoder, self).__init__()
            
        def forward(self, x):
            return torch.mul(1.,x)



class Phi_r(torch.nn.Module):
    def __init__(self):
        super(Phi_r, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

phi_r = Phi_r()

print('Phi Model type: '+genSuffixModel)
print(phi_r)
print('Number of trainable parameters = %d'%(sum(p.numel() for p in phi_r.parameters() if p.requires_grad)))






############################################Lightning Module#######################################################################
class HParam:
    def __init__(self):
        self.iter_update     = []
        self.nb_grad_update  = []
        self.lr_update       = []
        self.n_grad          = 1
        self.dim_grad_solver = 10
        self.dropout         = 0.25
        self.w_loss          = []
        self.automatic_optimization = True
        self.k_batch         = 1
        self.flag_uv_param = "u-v"
        
        self.GradType       = 1 
        self.OptimType      = 2 

        self.alpha_proj    = 0.5
        self.alpha_sr      = 0.5
        self.alpha_lr      = 0.5  # 1e4
        self.alpha_mse_ssh = 10.
        self.alpha_mse_gssh = 1.
        #self.alpha_mse_uv = 1.

        self.alpha_mse_vv = 1.
        self.flag_median_output = False
        self.median_filter_width = width_med_filt_spatial
        self.dw_loss = 32

        self.p_norm_loss = 2. 
        self.q_norm_loss = 2. 
        self.r_norm_loss = 2. 
        self.thr_norm_loss = 2.

        self.flagNoSSTObs = False
        
                          
        self.alpha          = np.array([1.,0.1])
        self.alpha4DVar     = np.array([0.01,1.])#np.array([0.30,1.60])#

        self.flagLearnWithObsOnly = False #True # 
        self.lambda_LRAE          = 0.5 # 0.5

        self.GradType       = 1 # Gradient computation (0: subgradient, 1: true gradient/autograd)
        self.OptimType      = 2 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
        
        self.NbProjection   = [0,0,0,0,0,0,0]#[0,0,0,0,0,0]#[5,5,5,5,5]##
        self.NBProjCurrent =0
         

       
class LitModel(pl.LightningModule):
    def __init__(self,conf=HParam(),*args, **kwargs):
        super().__init__()
        
        
        # hyperparameters
        self.hparams.alpha          = np.array([1.,0.1])
        self.hparams.alpha4DVar     = np.array([0.01,1.])#np.array([0.30,1.60])#

        self.hparams.flagLearnWithObsOnly = False #True # 
        self.hparams.lambda_LRAE          = 0.5 # 0.5

        self.hparams.GradType       = 1 # Gradient computation (0: subgradient, 1: true gradient/autograd)
        self.hparams.OptimType      = 2 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
        
        self.hparams.NbProjection   = [0,0,0,0,0,0,0]#[0,0,0,0,0,0]#[5,5,5,5,5]##
        self.hparams.iter_update     = [0,30,100,150,200,250,400]  # [0,2,4,6,9,15]
        self.hparams.nb_grad_update  = [5,5,10,15,20,20,20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.hparams.lr_update       = [1e-3,1e-4,1e-4,1e-4,1e-4,1e-5,1e-6,1e-6,1e-7]
        self.hparams.k_batch         = 1
        
         
        self.hparams.n_grad          = self.hparams.nb_grad_update[0]
        
        self.hparams.alpha_proj    = 0.5
        self.hparams.alpha_sr      = 0.5
        self.hparams.alpha_lr      = 0.5  # 1e4
        self.hparams.alpha_mse_ssh = 10.
        self.hparams.alpha_mse_gssh = 1.
        self.hparams.alpha_mse_vv = 1.
        
        self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(w_), requires_grad=False)
        self.hparams.automatic_optimization = False#True#

        
        self.hparams.NBProjCurrent =self.hparams.NbProjection[0]
        
        #lrCurrent       = lrUpdate[0] ??
            
        # main model
        self.model           = NN_4DVar.Model_4DVarNN_GradFP(phi_r,shapeData,self.hparams.NBProjCurrent,self.hparams.n_grad,self.hparams.GradType,self.hparams.OptimType,InterpFlag,UsePriodicBoundary)                
        self.save_hyperparameters()

        self.w_loss       = self.hparams.w_loss # duplicate for automatic upload to gpu
        self.x_pred        = None # variable to store output of test method
        
        self.automatic_optimization = self.hparams.automatic_optimization
        self.curr = 0
        
    def forward(self):
        return 1

    def configure_optimizers(self):
        #optimizer = optim.Adam(self.model.parameters(), lr= self.lrUpdate[0])
        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(),'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': self.hparams.lr_update[0]}
                                    ], lr=0.)       
        return optimizer
    
    def on_epoch_start(self):
        # enfore acnd check some hyperparameters 
        self.model.n_grad   = self.hparams.n_grad 
        
    
        
        
    def on_train_epoch_start(self):
        opt = self.optimizers()
        if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
            indx             = self.hparams.iter_update.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f'%(self.current_epoch,self.hparams.nb_grad_update[indx],self.hparams.lr_update[indx]))
            
            self.hparams.n_grad = self.hparams.nb_grad_update[indx]
            self.model.n_grad   = self.hparams.n_grad 
            
            mm = 0
            lrCurrent = self.hparams.lr_update[indx]
            lr = np.array([lrCurrent,lrCurrent,lrCurrent,0.5*lrCurrent,lrCurrent,0.])            
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]# * self.hparams.learning_rate
                mm += 1
                
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        opt = self.optimizers()
                    
        # compute loss and metrics    
        loss, out, metrics = self.compute_loss(train_batch, phase='train')

        # log step metric        
        #self.log('train_mse', mse)
        #self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] / var_Tr , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("tr_mse", metrics['mse_uv'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("tr_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # initial grad value
        if self.hparams.automatic_optimization == False :
            # backward
            self.manual_backward(loss)
        
            if (batch_idx + 1) % self.hparams.k_batch == 0:
                # optimisation step
                opt.step()
                
                # grad initialization to zero
                opt.zero_grad()
         
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')

        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse'] / var_Val , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_uv", metrics['mse_uv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics = self.compute_loss(test_batch, phase='test')

        self.log('test_loss', loss)
        self.log("test_mse", metrics['mse'] / var_Tt , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("test_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("test_vv", metrics['mse_vv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        out_debit = out
        return {'preds_debit': out_debit.detach().cpu()}

    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs
        print('.. \n')
    
    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds_debit'] for chunk in outputs]).numpy()
        x_test_rec = stdTrtrue * x_test_rec + meanTrtrue        
        self.x_rec_debit = x_test_rec[:,0,:,int(DT/2)]
        print(self.x_rec_debit.shape)

        #x_test_debit_obs = torch.cat([chunk['obs_debit'] for chunk in outputs]).numpy()
        #x_test_debit_obs[ x_test_debit_obs == 0. ] = np.float('NaN')
        #x_test_debit_obs = stdTr * x_test_debit_obs + meanTr
        #self.x_rec_debit_obs = x_test_debit_obs[:,:,:,int(DT/2)]

        return 1.

    def compute_loss(self, batch, phase):

        inputs_init,inputs_missing,masks,targets_GT = batch
        #inputs_init shape :batch_size*Nb_stations*(DT-1)
        # reshaping tensors
        inputs_init    = inputs_init.view(-1,1,inputs_init.size(1),inputs_init.size(2))
        inputs_missing = inputs_missing.view(-1,1,inputs_init.size(2),inputs_init.size(3))
        masks          = masks.view(-1,1,inputs_init.size(2),inputs_init.size(3))
        targets_GT     = targets_GT.view(-1,1,inputs_init.size(2),inputs_init.size(3))
        
        
       # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, masks,None,None)

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()
                       
            loss_R      = torch.sum((outputs - targets_GT)**2 * masks )
            loss_R      = torch.mul(1.0 / torch.sum(masks),loss_R)
            loss_I      = torch.sum((outputs - targets_GT)**2 * (1. - masks) )
            loss_I      = torch.mul(1.0 / torch.sum(1.-masks),loss_I)
            loss_All    = torch.mean((outputs - targets_GT)**2 )
            loss_AE     = torch.mean((self.model.phi_r(outputs) - outputs)**2 )
            loss_AE_GT  = torch.mean((self.model.phi_r(targets_GT) - targets_GT)**2 )
            
           
            # total loss
            loss        = self.hparams.alpha[0] * loss_All + 0.5 * self.hparams.alpha[1] * ( loss_AE + loss_AE_GT )
            
            # metrics
            mse = loss_All.detach()
            metrics   = dict([('mse',mse)])
            #print(mse.cpu().detach().numpy())
            
            outputs = outputs
        return loss,outputs, metrics
    

 
'''

def save_NetCDF(saved_path1, I_test, debit_gt , debit_obs, debit_rec ):
    
    indStat=31
    Stations = np.arange(1,32)
    print(len(Stations))
 
    #indN_Tt = np.concatenate([np.arange(60, 80)])
    indN_Tt = np.concatenate([np.arange(I_test[0][0]+int((DT+1)/2),I_test[0][1]-int((DT+1)/2))])
    
    for k in I_test[1::] :
        Inter=np.concatenate([np.arange(k[0]+int((DT+1)/2),k[1]-int((DT+1)/2))])
        indN_Tt=np.concatenate((indN_Tt,Inter))

    time_ = [datetime.datetime.strftime(datetime.datetime.strptime("1960-01-01", '%Y-%m-%d') \
                                       + datetime.timedelta(days=np.float64(i)), "%Y-%m-%d") for i in indN_Tt]
  
    xrdata = xr.Dataset( \
                        data_vars={'stations': (('Stations'), Stations), \
                   'Time': (('time'), time_), \
                   'debit_gt': (('time', 'Stations'), debit_gt), \
                   'debit_obs': (('time', 'Stations'), debit_obs), \
                   'debit_rec': (('time', 'Stations'), debit_rec)}, \
        coords={'S': Stations,  'time': indN_Tt},)
    xrdata.time.attrs['units'] = 'days since 1960-01-01 00:00:00'
    xrdata.to_netcdf(path=saved_path1, mode='w')
   

def compute_metrics(X_test,X_rec):
    # MSE
    print(X_test.shape)
    print(X_rec.shape)
    print(type(X_rec))
    print(((X_test - X_rec)**2).shape)
    M=np.mean((X_test - X_rec)**2,0)
    print(M.shape)
    mse = np.mean( (X_test - X_rec)**2 )
    
    return {'mse':mse}

def rmse_based_scores(ds_oi, ds_ref):
    
    # RMSE(t) based score
    rmse_t = 1.0 - (((ds_oi['sossheig'] - ds_ref['sossheig'])**2).mean(dim=('lon', 'lat')))**0.5/(((ds_ref['sossheig'])**2).mean(dim=('lon', 'lat')))**0.5
    # RMSE(x, y) based score
    # rmse_xy = 1.0 - (((ds_oi['sossheig'] - ds_ref['sossheig'])**2).mean(dim=('time')))**0.5/(((ds_ref['sossheig'])**2).mean(dim=('time')))**0.5
    rmse_xy = (((ds_oi['sossheig'] - ds_ref['sossheig'])**2).mean(dim=('time')))**0.5
    
    rmse_t = rmse_t.rename('rmse_t')
    rmse_xy = rmse_xy.rename('rmse_xy')

    # Temporal stability of the error
    reconstruction_error_stability_metric = rmse_t.std().values

    # Show leaderboard SSH-RMSE metric (spatially and time averaged normalized RMSE)
    leaderboard_rmse = 1.0 - (((ds_oi['sossheig'] - ds_ref['sossheig']) ** 2).mean()) ** 0.5 / (
        ((ds_ref['sossheig']) ** 2).mean()) ** 0.5

    
    return rmse_t, rmse_xy, np.round(leaderboard_rmse.values, 3), np.round(reconstruction_error_stability_metric, 3)



def psd_based_scores(ds_oi, ds_ref):
            
    # Compute error = SSH_reconstruction - SSH_true
    err = (ds_oi['sossheig'] - ds_ref['sossheig'])
    err = err.chunk({"lat":1, 'time': err['time'].size, 'lon': err['lon'].size})
    
    # make time vector in days units 
    err['time'] = (err.time - err.time[0]) #/ numpy.timedelta64(1, 'D')
    
    # Rechunk SSH_true
    signal = ds_ref['sossheig'].chunk({"lat":1, 'time': ds_ref['time'].size, 'lon': ds_ref['lon'].size})
    # make time vector in days units
    signal['time'] = (signal.time - signal.time[0]) #/ numpy.timedelta64(1, 'D')

    # Compute PSD_err and PSD_signal
    psd_err = xrft.power_spectrum(err, dim=['time', 'lon'], detrend='linear', window=True).compute()
    psd_signal = xrft.power_spectrum(signal, dim=['time', 'lon'], detrend='linear', window=True).compute()
    
    # Averaged over latitude
    mean_psd_signal = psd_signal.mean(dim='lat').where((psd_signal.freq_lon > 0.) & (psd_signal.freq_time > 0), drop=True)
    mean_psd_err = psd_err.mean(dim='lat').where((psd_err.freq_lon > 0.) & (psd_err.freq_time > 0), drop=True)
    
    # return PSD-based score
    psd_based_score = (1.0 - mean_psd_err/mean_psd_signal)

    # Find the key metrics: shortest temporal & spatial scales resolved based on the 0.5 contour criterion of the PSD_score 

    level = [0.5]    
    cs = plt.contour(1./psd_based_score.freq_time.values,1./psd_based_score.freq_lon.values, psd_based_score, level)
    x05, y05 = cs.collections[0].get_paths()[0].vertices.T
    plt.close()
    
    shortest_spatial_wavelength_resolved = np.min(x05)
    shortest_temporal_wavelength_resolved = np.min(y05)

    return (1.0 - mean_psd_err/mean_psd_signal), np.round(shortest_spatial_wavelength_resolved, 3), np.round(shortest_temporal_wavelength_resolved, 3)


if __name__ == '__main__':
    
    flagProcess = 1
    
    if flagProcess == 1: ## training model from scratch
    
        
        loadTrainedModel = 0#False#
        if loadTrainedModel == 1 :             
            
            pathCheckPOint = ''
            
            print('.... load pre-trained model :'+pathCheckPOint)
            mod = LitModel.load_from_checkpoint(pathCheckPOint)
            #mod.compute_graduv = Compute_graduv()
             
            mod.hparams.n_grad          = 5
            mod.hparams.iter_update     = [0, 20, 40, 60, 150, 150, 800]  # [0,2,4,6,9,15]
            mod.hparams.nb_grad_update  = [5, 5, 10, 10, 15, 20, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            mod.hparams.lr_update       = [1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-7]
        else:
            mod = LitModel()
       
                   
        filename_chkpt = 'model_Debit'
        
        print('..... Filename chkpt: '+filename_chkpt)
        
        print(mod.hparams)
        print('n_grad = %d'%mod.hparams.n_grad)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath= dirSAVE+'-'+suffix_exp,
                                              filename= filename_chkpt + '-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=3,
                                              mode='min')
        profiler_kwargs = {'max_epochs': 300 }

        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1, **profiler_kwargs,callbacks=[checkpoint_callback])
    
        ## training loop
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        X_val= []
        for i in IndVal:
            X_val.append(dataset[i[0]:i[1]])                
        val_mseRec = compute_metrics(X_val,mod.x_rec)         
        
        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        #ncfile = Dataset("results/test.nc","r")
        #X_rec  = ncfile.variables['ssh'][:]
        #ncfile.close()
        X_test= []
        for i in IndTest:
            X_test.append(dataset[i[0]:i[1]])
                         
        test_mseRec = compute_metrics(X_test,mod.x_rec)     
        
        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (Debit) :  -- 4DVarNN = %.3e  %%'%(val_mseRec['mse']))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) :  -- 4DVarNN = %.3e %%'%(test_mseRec['mse']))
        

    elif flagProcess == 3: ## test trained model with the Lightning code

        
        pathCheckPOint = 'ResDanube4DVar/-exp2/model_Debit-epoch=242-val_loss=0.00.ckpt'
        mod = LitModel.load_from_checkpoint(pathCheckPOint)     
        
        mod.hparams.n_grad = 5
        mod.hparams.median_filter_width = 1
        mod.hparams.flag_median_output = False
        width_med_filt_spatial = 1
        width_med_filt_temp = 1

        profiler_kwargs = {'max_epochs': 200}
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
        X_val    =  dataValNoNaND[:,:,int(DT/2)]
        
        ## postprocessing
        #if width_med_filt_spatial + width_med_filt_temp > 2 :
            #mod.x_rec_debit = ndimage.median_filter(mod.x_rec_debit,size=(width_med_filt_temp,width_med_filt_spatial,width_med_filt_spatial))
        
        val_mseRec = compute_metrics(X_val,mod.x_rec_debit)     
       
        val_norm_rmse_debit = np.sqrt( np.mean( X_val.ravel()**2 ) )

        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
                         
        X_Test   =  dataTestNoNaND[:,:,int(DT/2)]
        #X_mask = []
        
        test_mseRec = compute_metrics(X_Test,mod.x_rec_debit)     
        
        
        test_norm_rmse_debit = np.sqrt( np.mean( X_Test.ravel()**2 ) )

        saveRes = True# 
        debit_gt = X_Test
        
        debit_obs = X_test_missingD[:,:,int(DT/2)]
        
        debit_rec = mod.x_rec_debit            
        
        if saveRes == True :
            
            filename_res = pathCheckPOint.replace('.ckpt','_res.nc')
            print('.... save all gt/rec fields in nc file '+filename_res)
            save_NetCDF(filename_res, 
                        Indtest,
                        debit_gt = debit_gt , 
                        debit_obs = debit_obs,
                        debit_rec = debit_rec
                    )

        print('...  model: '+pathCheckPOint)
       
        
        print('....................................')
        print('....... Validation dataset')
        print('....... NRMSE Val dataset (Debit) :  -- 4DVarNN = %.3f %%'%(1.-np.sqrt(val_mseRec['mse'])/val_norm_rmse_debit))
        print('....... MSE Val dataset (Debit) :  -- 4DVarNN = %.3e   %%'%(val_mseRec['mse']))

        print(' ')
        print('....... Test dataset')
        print('....... NRMSE Test dataset (Debit) : -- 4DVarNN = %.3f %%'%(1.-np.sqrt(test_mseRec['mse'])/test_norm_rmse_debit))
        print('....... MSE Test dataset (Debit) :  -- 4DVarNN = %.3e  %%'%(test_mseRec['mse']))
        
       