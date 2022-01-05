#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:59:23 2020
@author: rfablet
"""

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
import torch_4DVarNN_dinAE_Copy1 as NN_4DVar
#import torch4DVarNN_solver as NN_4DVar
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from netCDF4 import Dataset
from statsmodels.distributions.empirical_distribution import ECDF



############################
'''
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import argparse
import numpy as np
import datetime

import matplotlib.pyplot as plt

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR


import os
import sys
sys.path.append('../4dvarnet-core')
import solver as NN_4DVar

from sklearn.feature_extraction import image

from scipy import ndimage

'''
parser = argparse.ArgumentParser()
flagProcess    = [0,1,2,3,4]#Sequence fo processes to be run
    
flagRandomSeed = 0
flagSaveModel  = 1
     
batch_size  = 256#4#4#8#12#8#256#8 originellement 96

dirSAVE     = './ResDanube4DVar/'
suffix_exp='exp3'
genFilename = 'Debit_v11'
  
flagAEType = 2 # 0: L96 model, 1-2: GENN
DimAE      = 50#50#10#50
    
UsePriodicBoundary = True # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
InterpFlag         = False

NbDays          = 18244

time_step  = 1
DT = 13
sigNoise   = np.sqrt(2)
rateMissingData = 0.5#0.9
width_med_filt_spatial = 5
width_med_filt_temp = 1


# loss weghing wrt time
w_ = np.zeros(DT)
w_[int(DT / 2)] = 1.
wLoss = torch.Tensor(w_)

flagTypeMissData = 4

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

seuil_10 = np.zeros(31)
#for i in range(31) :
#    Si = sorted(L[i],reverse = True)[int(18244/10)]
#    print(Si)
#    seuil_10[i]=Si[0]
seuil_10 =np.array([2220.,  1020. ,  764.,   744. ,  577. ,  535.,   511.,   329.,   279. ,  218.,
   92.3 ,  45.,  1240.,   257. ,  250.,   116. ,   81.5,   46.7,   35.3,  179.,
  130.,   123.,    96.5,   78.6 ,  69.,    62.1,   47.6 ,  81.8,   74.4 , 449.,
  427. ])    
print(seuil_10)


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



"""


# Definiton of training, validation and test dataset
# from dayly indices over a one-year time series

suffix_exp = "exp2"
day_0 = datetime.date(2012,10,1)

if suffix_exp == "exp3" :
    iiVal = 60 - int(dT / 2)
    jjVal = iiVal + 20 + int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 10 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 10 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "exp2" :
    day_val  = datetime.date(2013,1,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    day_test_0  = datetime.date(2012,10,22)
    day_test_1  = datetime.date(2012,12,2)
    iiTest = int((day_test_0 - day_0).days) - int(dT / 2) #90 - int(dT / 2)
    jjTest = int((day_test_1 - day_0).days) + int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = jjVal + 10
    jjTr1 = 365
    
    iiTr2 = 365 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "summer" : # Summer
    day_val  = datetime.date(2013,7,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 10 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 10 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "spring" : # Spring
    day_val  = datetime.date(2013,4,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 10 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 10 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "winter" : # Winter
    day_val  = datetime.date(2013,1,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 10 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 10 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "fall" : # Fall
    day_val  = datetime.date(2012,10,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 10 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 10 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "winter2" : # Winter
    day_val  = datetime.date(2013,1,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 30 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 30 #130 + int(dT / 2)
    jjTr2 = 365
elif suffix_exp == "spring2" : # Spring
    day_val  = datetime.date(2013,4,1)
    iiVal = int((day_val - day_0).days)
    jjVal = iiVal + 20 + 2*int(dT / 2) #int(dT / 2)
    
    iiTest = jjVal + 30 #90 - int(dT / 2)
    jjTest = iiTest + 20 + 2*int(dT / 2) # 110 + int(dT / 2)
     
    iiTr1 = 0
    jjTr1 = iiVal - 10 #50 - int(dT / 2)
    
    iiTr2 = jjTest + 30 #130 + int(dT / 2)
    jjTr2 = 365
"""

    
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
#flagTpeMissData = 4  : Prevision

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
elif flagTypeMissData == 3 :
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
    
else : 
    dataTrainingD    = np.zeros((dataTrainingNoNaND.shape))
    dataTrainingD[:] = float('nan')
            
    dataValD    = np.zeros((dataValNoNaND.shape))
    dataValD[:] = float('nan')
            
    dataTestD        = np.zeros((dataTestNoNaND.shape))
    dataTestD[:]     = float('nan')
    
    dataTrainingD[:,:(int(2*DT/3+1)),:] = dataTrainingNoNaND[:,:(int(2*DT/3+1)),:]
    dataValD[:,:(int(2*DT/3+1)),:] = dataValNoNaND[:,:(int(2*DT/3+1)),:]
    dataTestD[:,:(int(2*DT/3+1)),:] = dataTestNoNaND[:,:(int(2*DT/3+1)),:]
    genSuffixObs    = '_ObsSubRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
    
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
            
elif flagTypeMissData == 4 :
    #Moyenne des débits par station sur l'échatillon total
    M = np.mean(X_trainD,0)
    mean_X_trainD = np.mean(M,1)
    mean_X_trainD = mean_X_trainD.reshape(31,1)
    meanTr        = mean_X_trainD
    meanTrtrue = meanTr
    #Moyenne des débits
    X_nomask=X_trainD[:,:,:(int(2*DT/3+1))]
    M2 = np.mean(X_nomask,0)
    mean_X_train_missingD = np.mean(M2,1)
    mean_X_train_missingD = mean_X_train_missingD.reshape(31,1)

    #Ecart-type:
    meansquaretrue = np.mean( (X_trainD-mean_X_trainD)**2,0)
    meansquare2true = np.mean(meansquaretrue,1)
    meansquare2true=meansquare2true.reshape(31,1)
    stdTrtrue           = np.sqrt(meansquare2true )
    
    #Normalisation et standardisation des données
    x_train_missingD = np.zeros(X_train_missingD.shape)
    x_train_missingD[:,:,:(int(2*DT/3)+1)]=(X_train_missingD[:,:,:(int(2*DT/3)+1)]-mean_X_trainD)/stdTrtrue
 
    x_val_missingD = np.zeros(X_val_missingD.shape)
    x_val_missingD[:,:,:(int(2*DT/3)+1)]=(X_val_missingD[:,:,:(int(2*DT/3)+1)]-mean_X_trainD)/stdTrtrue

    x_test_missingD = np.zeros(X_test_missingD.shape)
    x_test_missingD[:,:,:(int(2*DT/3)+1)]=(X_test_missingD[:,:,:(int(2*DT/3)+1)]-mean_X_trainD)/stdTrtrue


    x_trainD = (X_trainD - mean_X_trainD) / stdTrtrue
    x_valD = (X_valD - mean_X_trainD) / stdTrtrue
    x_testD  = (X_testD - mean_X_trainD) / stdTrtrue
    
    stdTr =stdTrtrue
    
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

m_seuil = ((seuil_10.reshape(31,1)-meanTr)/stdTr)
print(seuil_10.shape)
print(meanTr.shape)
print(stdTr.shape)
print("seuil normalisé")
print(m_seuil)
        
print('..... Training dataset: %dx%dx%d'%(x_train_missingD.shape[0],x_trainD.shape[1],x_trainD.shape[2]))
print('..... Validation dataset: %dx%dx%d'%(x_valD.shape[0],x_valD.shape[1],x_valD.shape[2]))
print('..... Test dataset    : %dx%dx%d'%(x_testD.shape[0],x_testD.shape[1],x_testD.shape[2]))
            

print('........ Initialize interpolated states')
## Initial interpolation
#flagInit = 0 : Masked values are replaced by 0
#flagInit = 1 : masked values are replaced by last available value (prevision)
#flaginit = 2 : Interpolation 

flagInit = 1

            
if flagInit == 0: 
    X_train_InitD = mask_trainD * X_train_obsD + (1. - mask_trainD) * (np.zeros(X_train_missingD.shape) + meanTr)
    X_val_InitD = mask_valD * X_val_obsD + (1. - mask_valD) * (np.zeros(X_val_missingD.shape) + meanTr)
    X_test_InitD  = mask_testD * X_test_obsD + (1. - mask_testD) * (np.zeros(X_test_missingD.shape) + meanTr)
    
elif flagInit==1 :
    X_ext_train = X_train_missingD[:,:,int(2*DT/3)].reshape(X_train_missingD.shape[0],X_train_missingD.shape[1],1)
    X_train_InitD = mask_trainD * X_train_obsD + (1. - mask_trainD)*X_ext_train
    X_ext_val = X_val_missingD[:,:,int(2*DT/3)].reshape(X_val_missingD.shape[0],X_val_missingD.shape[1],1)
    X_val_InitD = mask_valD * X_val_obsD + (1. - mask_valD)*X_ext_val
    X_ext_test = X_test_missingD[:,:,int(2*DT/3)].reshape(X_test_missingD.shape[0],X_test_missingD.shape[1],1)
    X_test_InitD = mask_testD * X_test_obsD + (1. - mask_testD)*X_ext_test
    
    
    
    
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
######################### data loaders
training_dataset   = torch.utils.data.TensorDataset(torch.Tensor(x_trainOI),torch.Tensor(x_trainObs),torch.Tensor(x_trainMask),torch.Tensor(ySST_train),torch.Tensor(x_train),torch.Tensor(u_train),torch.Tensor(v_train),torch.Tensor(vv_train)) # create your datset
val_dataset        = torch.utils.data.TensorDataset(torch.Tensor(x_valOI),torch.Tensor(x_valObs),torch.Tensor(x_valMask),torch.Tensor(ySST_val),torch.Tensor(x_val),torch.Tensor(u_val),torch.Tensor(v_val),torch.Tensor(vv_val)) # create your datset
test_dataset       = torch.utils.data.TensorDataset(torch.Tensor(x_testOI),torch.Tensor(x_testObs),torch.Tensor(x_testMask),torch.Tensor(ySST_test),torch.Tensor(x_test),torch.Tensor(u_test),torch.Tensor(v_test),torch.Tensor(vv_test))  # create your datset

dataloaders = {
    'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
}            

var_Tr    = np.var( x_train )
var_Tt    = np.var( x_test )
var_Val   = np.var( x_val )
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



def L_hat_opti(u,Y,X):
    Ind = torch.where(X>u)
    if len(Ind[0])==0 :
        return(1)
    else : 
        W=Y.clone().detach().cpu()
        ecdf = ECDF(W.numpy())
        l=len(W.numpy())
        def G_thilde_opti(x):
            return(l/(l+1)*(1-ecdf(x))+1/(l+1))
        G_u= G_thilde_opti(u)
        M= X[Ind].clone().detach().cpu()
        B= G_thilde_opti(M.numpy())/G_u
        C=torch.from_numpy(B)
        A = torch.log(C)
        res=torch.sum(A)/len(Ind[0])+1 
        if np.isnan(res):
            print(len(Ind))
            print(Ind)
            print(G_u)
            print(M)
            print(B)
            print(A)
        return(res)


def K_hat(u,Y,X):
    return(-L_hat_opti(u,Y,X)-L_hat_opti(u,X,Y))


def K_hat_tensor(seuil,X,Y):
    n=X.shape[0]
    res=0
    for i in range(n):
        resi=K_hat(seuil[i],X[i,:],Y[i,:])
        res+=resi
    return(res)


def save_NetCDF(saved_path1, I_test, debit_gt , debit_obs, debit_rec, mode ):
    
    if mode == 'reconstruction' :
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
    
    else : 
        Stations = np.arange(1,32)
        print(len(Stations))
        NbDays = DT
 
        #indN_Tt = np.concatenate([np.arange(60, 80)])
        indN_Tt = np.concatenate([np.arange(I_test[0][0]+int((DT+1)/2),I_test[0][1]-int((DT+1)/2))])
        for k in I_test[1::] :
            Inter=np.concatenate([np.arange(k[0]+int((DT+1)/2),k[1]-int((DT+1)/2))])
            indN_Tt=np.concatenate((indN_Tt,Inter))

        time_ = [datetime.datetime.strftime(datetime.datetime.strptime("1960-01-01", '%Y-%m-%d') \
                                       + datetime.timedelta(days=np.float64(i)), "%Y-%m-%d") for i in indN_Tt]
        Days=np.arange(1,NbDays+1)

        xrdata = xr.Dataset( data_vars={'stations': (('Stations'), Stations),'Time': (('time'), time_), \
                   'Day':(('day'),Days), \
                   'debit_gt': (('time', 'Stations','day'), debit_gt), \
                   'debit_obs': (('time', 'Stations','day'), debit_obs), \
                   'debit_rec': (('time', 'Stations','day'), debit_rec)}, \
        coords={'S': Stations,  'time': indN_Tt, 'D':Days},)
        xrdata.time.attrs['units'] = 'days since 1960-01-01 00:00:00'
        xrdata.to_netcdf(path=saved_path1, mode='w')
'''
def save_NetCDF(saved_path1, ind_start,ind_end, ssh_gt , ssh_oi, ssh_obs, sst_gt, u_gt, v_gt, vv_gt, ssh_rec, u_rec_geo, v_rec_geo, u_rec, v_rec, vv_rec, vv_rec_swot, feat_sst ):
    
    extent = [-65., -55., 30., 40.]
    indLat = 200
    indLon = 200

    lon = np.arange(extent[0], extent[1], 1 / (20 / dwscale))
    lat = np.arange(extent[2], extent[3], 1 / (20 / dwscale))
    indLat = int(indLat / dwscale)
    indLon = int(indLon / dwscale)
    lon = lon[:indLon]
    lat = lat[:indLat]

    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T

    #indN_Tt = np.concatenate([np.arange(60, 80)])
    indN_Tt = np.concatenate([np.arange(ind_start, ind_end)])
    time_ = [datetime.datetime.strftime(datetime.datetime.strptime("2012-10-01", '%Y-%m-%d') \
                                       + datetime.timedelta(days=np.float64(i)), "%Y-%m-%d") for i in indN_Tt]

    xrdata = xr.Dataset( \
        data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                   'latitude': (('lat', 'lon'), mesh_lat), \
                   'Time': (('time'), time_), \
                   'ssh_gt': (('time', 'lat', 'lon'), ssh_gt), \
                   'ssh_oi': (('time', 'lat', 'lon'), ssh_oi), \
                   'ssh_obs': (('time', 'lat', 'lon'), ssh_obs), \
                   'sst_gt': (('time', 'lat', 'lon'), sst_gt), \
                   'u_gt': (('time', 'lat_uv', 'lon_uv'), u_gt), \
                   'v_gt': (('time', 'lat_uv', 'lon_uv'), v_gt), \
                   'vv_gt': (('time', 'lat_uv', 'lon_uv'), vv_gt), \
                   'ssh_rec': (('time', 'lat', 'lon'), ssh_rec), \
                   'u_rec_geo': (('time', 'lat_uv', 'lon_uv'), u_rec_geo), \
                   'v_rec_geo': (('time', 'lat_uv', 'lon_uv'), v_rec_geo), \
                   'u_rec': (('time', 'lat_uv', 'lon_uv'), u_rec), \
                   'v_rec': (('time', 'lat_uv', 'lon_uv'), v_rec), \
                   'vv_rec': (('time', 'lat_uv', 'lon_uv'), vv_rec), \
                   'vv_rec_swot': (('time', 'lat_uv', 'lon_uv'), vv_rec), \
                   'feat_sst': (('time','dT','lat', 'lon'), feat_sst)}, \
        coords={'lon': lon, 'lat': lat,'dT': np.arange(0,feat_sst.shape[1]),'lon_uv': lon[1:-1], 'lat_uv': lat[1:-1], 'time': indN_Tt},)
    xrdata.time.attrs['units'] = 'days since 2012-10-01 00:00:00'
    xrdata.to_netcdf(path=saved_path1, mode='w')
    
'''    

'''
#######################################Phi_r, Model_H, Model_Sampling architectures ################################################

print('........ Define AE architecture')
shapeData      = np.array(x_train.shape[1:])
shapeData_test = np.array(x_test.shape[1:])
if flag_aug_state == 1 :
    shapeData[0]  += 3*shapeData[0]
elif flag_aug_state == 2 :
    shapeData[0]  += 4*shapeData[0]
else:
    shapeData[0]  += 2*shapeData[0]

shapeDataSST   = np.array(ySST_train.shape[1:])

dW  = 3
dW2 = 1
sS  = int(4/dx)
nbBlocks = 1
rateDr   = 0. * rateDropout

class BiLinUnit(torch.nn.Module):
    def __init__(self,dimIn,dim,dropout=0.):
        super(BiLinUnit, self).__init__()
        
        self.conv1  = torch.nn.Conv2d(dimIn, 2*dim, (2*dW+1,2*dW+1),padding=dW, bias=False)
        self.conv2  = torch.nn.Conv2d(2*dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.conv3  = torch.nn.Conv2d(2*dim, dimIn, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.bilin0 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.bilin1 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.bilin2 = torch.nn.Conv2d(dim, dim, (2*dW2+1,2*dW2+1), padding=dW2, bias=False)
        self.dropout  = torch.nn.Dropout(dropout)
        
    def forward(self,xin):
        
        x = self.conv1(xin)
        x = self.dropout(x)
        x = self.conv2( F.relu(x) )
        x = self.dropout(x)
        x = torch.cat((self.bilin0(x), self.bilin1(x) * self.bilin2(x)),dim=1)
        x = self.dropout(x)
        x = self.conv3( x )
        
        return x
   
class Encoder(torch.nn.Module):
    def __init__(self,dimInp,dimAE,rateDropout=0.):
        super(Encoder, self).__init__()
        self.NbBlocks  = nbBlocks
        self.DimAE     = dimAE
        #self.conv1HR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        #self.conv1LR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False) 
        self.pool1   = torch.nn.AvgPool2d(sS)
        self.convTr  = torch.nn.ConvTranspose2d(dimInp,dimInp,(sS,sS),stride=(sS,sS),bias=False)          

        #self.NNtLR    = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)
        #self.NNHR     = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)                      
        self.NNLR     = self.__make_BilinNN(dimInp,self.DimAE,self.NbBlocks,rateDropout)
        self.NNHR     = self.__make_BilinNN(dimInp,self.DimAE,self.NbBlocks,rateDropout)                      
        self.dropout  = torch.nn.Dropout(rateDropout)
      
    def __make_BilinNN(self,dimInp,dimAE,Nb_Blocks=2,dropout=0.): 
          layers = []
          layers.append( BiLinUnit(dimInp,dimAE,dropout) )
          for kk in range(0,Nb_Blocks-1):
              layers.append( BiLinUnit(dimAE,dimAE,dropout) )
          return torch.nn.Sequential(*layers)
      
    def forward(self, xinp):
        
        ## LR comlponent
        xLR = self.NNLR( self.pool1(xinp) )
        xLR = self.dropout(xLR)
        xLR = self.convTr( xLR ) 
        
        # HR component
        xHR = self.NNHR( xinp )
        
        return xLR + xHR
  class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
  
    def forward(self, x):
        return torch.mul(1.,x)


class Phi_r(torch.nn.Module):
    def __init__(self):
        super(Phi_r, self).__init__()
        self.encoder = Encoder(shapeData[0],DimAE,rateDr)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

phi_r = Phi_r()

print(phi_r)
print('Number of trainable parameters = %d' % (sum(p.numel() for p in phi_r.parameters() if p.requires_grad)))
W_KERNEL_MODEL_H = 5

class Model_H(torch.nn.Module):
    def __init__(self,width_kernel=3,dim=5):
        super(Model_H, self).__init__()

        self.DimObs = 2
        self.w_kernel = width_kernel
        self.dimObsChannel = np.array([shapeData[0],dim])

        self.conv11  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False)

        self.conv12  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False)
        self.conv22  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False)
                    
        self.conv21  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(self.w_kernel,self.w_kernel),padding=int(self.w_kernel/2),bias=False)
        self.convM   = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(3,3),padding=int(self.w_kernel/2),bias=False)
        self.S       = torch.nn.Sigmoid()#torch.nn.Softmax(dim=1)

    def forward(self, x , y , mask):
        dyout  = (x - y[0]) * mask[0] 
        
        y1     = y[1] * mask[1]
        dyout1 = self.conv11(x) - self.conv21(y1)
        dyout1 = dyout1 * self.S( self.convM( mask[1] ) )                  
        
        return [dyout,dyout1]

class Gradient_img(torch.nn.Module):
    def __init__(self):
        super(Gradient_img, self).__init__()

        a = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

        b = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, im):

        if im.size(1) == 1:
            G_x = self.convGx(im)
            G_y = self.convGy(im)
            G = torch.sqrt(torch.pow(0.5 * G_x, 2) + torch.pow(0.5 * G_y, 2))
        else:

            for kk in range(0, im.size(1)):
                G_x_ = self.convGx(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))
                G_y_ = self.convGy(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))

                G_x_ = G_x_.view(-1, 1, im.size(2) - 2, im.size(2) - 2)
                G_y_ = G_y_.view(-1, 1, im.size(2) - 2, im.size(2) - 2)
                nG_ = torch.sqrt(torch.pow(0.5 * G_x_, 2) + torch.pow(0.5 * G_y_, 2))

                if kk == 0:
                    nG = nG_.view(-1, 1, im.size(1) - 2, im.size(2) - 2)
                    Gx = G_x_.view(-1, 1, im.size(1) - 2, im.size(2) - 2)
                    Gy = G_y_.view(-1, 1, im.size(1) - 2, im.size(2) - 2)
                else:
                    nG = torch.cat((nG, nG_.view(-1, 1, im.size(1) - 2, im.size(2) - 2)), dim=1)
                    Gx = torch.cat((Gx, G_x_.view(-1, 1, im.size(1) - 2, im.size(2) - 2)), dim=1)
                    Gy = torch.cat((Gy, G_y_.view(-1, 1, im.size(1) - 2, im.size(2) - 2)), dim=1)
        return nG,Gx,Gy
gradient_img = Gradient_img()

class Div_uv(torch.nn.Module):
    def __init__(self):
        super(Div_uv, self).__init__()

        a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

        b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0), requires_grad=False)

    def forward(self, u,v):

        if u.size(1) == 1:
            G_x = self.convGx(u)
            G_y = self.convGy(v)
        else:

            for kk in range(0, u.size(1)):
                G_x = self.convGx(u[:, kk, :, :].view(-1, 1, u.size(2), u.size(3)))
                G_y = self.convGy(v[:, kk, :, :].view(-1, 1, u.size(2), u.size(3)))

                G_x = G_x.view(-1, 1, u.size(2) - 2, u.size(2) - 2)
                G_y = G_y.view(-1, 1, u.size(2) - 2, u.size(2) - 2)

                div_ = G_x + G_y 
                if kk == 0:
                    div = div_.view(-1, 1, u.size(1) - 2, u.size(2) - 2)
                else:
                    div = torch.cat((div, div_.view(-1, 1, u.size(1) - 2, u.size(2) - 2)), dim=1)
        return div
    
model_div = Div_uv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_div = model_div.to(device)

class Compute_graduv(torch.nn.Module):
    def __init__(self):
        super(Compute_graduv, self).__init__()

        self.alpha = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.alpha.weight = torch.nn.Parameter(torch.from_numpy(np.array([1.0])).float().unsqueeze(0).unsqueeze(0).unsqueeze(0), requires_grad=False)
        
        #self.alpha_v = torch.nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.convG = torch.nn.Conv2d(1, 1, kernel_size=(3,3), stride=1, padding=(0,0), bias=False)
        
        if 1*1 :    
            a = 0.25*np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
            #self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            self.convG.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            #self.convGx = torch.nn.Conv2d(1, 1, kernel_size=(1,3), stride=1, padding=(0,1), bias=False)
            
    def forward(self, im, du=1.,dv=1.):
        for kk in range(0, im.size(1)):
            _g_u = self.alpha( self.convG(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3))) )
            _g_v = self.convG( torch.transpose( im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)) , 2 , 3 ) )
            _g_v = self.alpha( torch.transpose( _g_v , 2 , 3 ) )

            _g_u = _g_u.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
            _g_v = _g_v.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
            if kk == 0:
                d_u = _g_u.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                d_v = _g_v.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
            else:
                d_u = torch.cat((d_u, _g_u.view(-1, 1, im.size(2) - 2, im.size(3) - 2)), dim=1)
                d_v = torch.cat((d_v, _g_v.view(-1, 1, im.size(2) - 2, im.size(3) - 2)), dim=1)
        
        return [ du * d_u , dv * d_v ]

class ModelLR(torch.nn.Module):
    def __init__(self):
        super(ModelLR, self).__init__()

        self.pool = torch.nn.AvgPool2d((16, 16))

    def forward(self, im):
        return self.pool(im)

alpha_MSE     = 0.1
alpha_Proj    = 0.5
alpha_SR      = 0.5
alpha_LR      = 0.5  # 1e4

# loss weghing wrt time
w_ = np.zeros(dT)
w_[int(dT / 2)] = 1.
wLoss = torch.Tensor(w_)


# recompute the MSE for OI on training dataset

def save_NetCDF(saved_path1, ind_start,ind_end, ssh_gt , ssh_oi, ssh_obs, sst_gt, u_gt, v_gt, vv_gt, ssh_rec, u_rec_geo, v_rec_geo, u_rec, v_rec, vv_rec, vv_rec_swot, feat_sst ):
    
    extent = [-65., -55., 30., 40.]
    indLat = 200
    indLon = 200

    lon = np.arange(extent[0], extent[1], 1 / (20 / dwscale))
    lat = np.arange(extent[2], extent[3], 1 / (20 / dwscale))
    indLat = int(indLat / dwscale)
    indLon = int(indLon / dwscale)
    lon = lon[:indLon]
    lat = lat[:indLat]

    mesh_lat, mesh_lon = np.meshgrid(lat, lon)
    mesh_lat = mesh_lat.T
    mesh_lon = mesh_lon.T

    #indN_Tt = np.concatenate([np.arange(60, 80)])
    indN_Tt = np.concatenate([np.arange(ind_start, ind_end)])
    time_ = [datetime.datetime.strftime(datetime.datetime.strptime("2012-10-01", '%Y-%m-%d') \
                                       + datetime.timedelta(days=np.float64(i)), "%Y-%m-%d") for i in indN_Tt]

    xrdata = xr.Dataset( \
        data_vars={'longitude': (('lat', 'lon'), mesh_lon), \
                   'latitude': (('lat', 'lon'), mesh_lat), \
                   'Time': (('time'), time_), \
                   'ssh_gt': (('time', 'lat', 'lon'), ssh_gt), \
                   'ssh_oi': (('time', 'lat', 'lon'), ssh_oi), \
                   'ssh_obs': (('time', 'lat', 'lon'), ssh_obs), \
                   'sst_gt': (('time', 'lat', 'lon'), sst_gt), \
                   'u_gt': (('time', 'lat_uv', 'lon_uv'), u_gt), \
                   'v_gt': (('time', 'lat_uv', 'lon_uv'), v_gt), \
                   'vv_gt': (('time', 'lat_uv', 'lon_uv'), vv_gt), \
                   'ssh_rec': (('time', 'lat', 'lon'), ssh_rec), \
                   'u_rec_geo': (('time', 'lat_uv', 'lon_uv'), u_rec_geo), \
                   'v_rec_geo': (('time', 'lat_uv', 'lon_uv'), v_rec_geo), \
                   'u_rec': (('time', 'lat_uv', 'lon_uv'), u_rec), \
                   'v_rec': (('time', 'lat_uv', 'lon_uv'), v_rec), \
                   'vv_rec': (('time', 'lat_uv', 'lon_uv'), vv_rec), \
                   'vv_rec_swot': (('time', 'lat_uv', 'lon_uv'), vv_rec), \
                   'feat_sst': (('time','dT','lat', 'lon'), feat_sst)}, \
        coords={'lon': lon, 'lat': lat,'dT': np.arange(0,feat_sst.shape[1]),'lon_uv': lon[1:-1], 'lat_uv': lat[1:-1], 'time': indN_Tt},)
    xrdata.time.attrs['units'] = 'days since 2012-10-01 00:00:00'
    xrdata.to_netcdf(path=saved_path1, mode='w')


m_NormObs = NN_4DVar.Model_WeightedL2Norm()     
m_NormPhi = NN_4DVar.Model_WeightedL2Norm()

FLAG_TRAINABLE_NORM = True
def compute_Lpqr_numpy(x0,x1,p_=2.,q_=2.,r_=2.,thr_=0.,eps=1e-10):
    
    if 1*1 : 
        nx = ( np.sqrt( x0[0]**2 + x0[1]**2 ) > thr_ )
        dx = np.sqrt( (x0[0]-x1[0])**2 + (x0[1]-x1[1])**2 )
        x =  nx * dx
        loss_ = np.nansum( np.power( np.abs(eps + x), p_ )  , axis = 2)
        loss_ = np.sum( loss_ , axis = 1)
     
        N = np.sum( np.sum((~np.isnan(x)) * nx , axis = 2) , axis = 1)
        loss_ = np.power( eps + loss_ / N  , q_/p_ )       
        
        loss_ = np.mean( np.power( np.power( eps + loss_,1./q_) , r_) , axis = 0)
    else:
        nx = ( np.sqrt( x0[0]**2 + x0[1]**2 ) > thr_ )
        dx = np.sqrt( (x0[0]-x1[0])**2 + (x0[1]-x1[1])**2 ) / np.sqrt( (x0[0])**2 + (x0[1])**2 )
        x =  nx * dx
        loss_ = np.nansum( np.power( np.abs(eps + x), p_ )  , axis = 2)
        loss_ = np.sum( loss_ , axis = 1)
     
        N = np.sum( np.sum( (~np.isnan(x)) * nx , axis = 2) , axis = 1)
        loss_ = np.power( eps + loss_ / N  , q_/p_ )       
        
        loss_ = np.mean( np.power( np.power( eps + loss_,1./q_) , r_) , axis = 0)
    
    return loss_

def compute_WeightedLoss_Lpqr(x,w,p_=2.,q_=2.,r_=2.,eps=1e-10):

    loss_ = torch.nansum( torch.pow( torch.abs(eps + x), p_ )  , dim = 3)
    loss_ = torch.sum( loss_ , dim = 2)
 
    N = torch.sum( torch.sum(~torch.isnan(x) , dim = 3) , dim = 2)
    loss_ = torch.pow( eps + loss_ / N  , q_/p_ )       
    
    loss_ = torch.sum( loss_ * w.repeat(x.shape[0],1) , dim = 1 )
    loss_ = torch.mean( torch.pow(torch.pow( eps + loss_,1./q_) , r_) , dim = 0)
    
    return loss_

class Model_NomrLpqr(torch.nn.Module):
    def __init__(self,p=2.,q=2.,r=2.,trainable=False):
        super(Model_NomrLpqr, self).__init__()
        #self.p = torch.nn.Parameter(torch.Tensor([np.exp(p-1.)-1.]), requires_grad=True)
        #self.q = torch.nn.Parameter(torch.Tensor([np.exp(q-1.)-1.]), requires_grad=True)
        self.trainable = trainable

        if FLAG_TRAINABLE_NORM == True :
            self.p = torch.nn.Parameter(torch.Tensor([p]), requires_grad=True)
            self.q = torch.nn.Parameter(torch.Tensor([q]), requires_grad=True)
            self.r = torch.nn.Parameter(torch.Tensor([r]), requires_grad=True)
        else:
            self.p = torch.nn.Parameter(torch.Tensor([p]), requires_grad=False)
            self.q = torch.nn.Parameter(torch.Tensor([q]), requires_grad=False)
            self.r = torch.nn.Parameter(torch.Tensor([r]), requires_grad=False)

        self.eps = torch.nn.Parameter(torch.Tensor([1e-10]), requires_grad=False)
        
    def forward(self,x,w,eps):

        p_,q_,r_ = self.compute_pqr()
            
        loss_ = torch.nansum( torch.pow( torch.abs(self.eps + x), p_ )  , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
 
        N = torch.sum(~torch.isnan(x) , dim = 3)
        N = torch.sum( N , dim = 2)
        loss_ = torch.pow( self.eps + loss_ / N  , q_/p_ )       
        
        loss_ = torch.nansum( loss_ * w.repeat(x.shape[0],1) , dim = 1 )
        loss_ = torch.mean( torch.pow(torch.pow( self.eps + loss_,1./q_) , r_) , dim = 0)

        return loss_
    def compute_pqr(self):
        
        if 1*0 :
            p_ = 1. + torch.log( 1 + torch.abs(self.eps + self.p ) )
            q_ = 1. + torch.log( 1 + torch.abs(self.eps + self.q ) )
        else:
           p_ = torch.abs(self.eps + self.p )
           q_ = torch.abs(self.eps + self.q )
           r_ = torch.abs(self.eps + self.r )

        return p_,q_,r_

alpha_ssh2u  = None#42.20436766972647
alpha_ssh2v = None#77.99700321505073

if alpha_ssh2u is None or alpha_ssh2v is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    graduv = Compute_graduv().to(device)
    running_loss_GOI = 0.
    running_loss_OI = 0.
    num_loss = 0
    
    num_ssh2u = 0.
    denum_ssh2u = 0.
    num_ssh2v = 0.
    denum_ssh2v = 0.

    for ssh_OI, inputs_obs,inputs_Mask, inputs_SST, ssh_gt, u_gt, v_gt, vv_gt in dataloaders['train']:
        ssh_gt   = ssh_gt.to(device)
        u_gt     = u_gt.to(device)
        v_gt     = v_gt.to(device)
        
        # gradient norm field
        g_u, g_v = graduv(ssh_gt)
        
        num_ssh2u += torch.sum( -1. * g_v * u_gt[:,:,1:-1,1:-1] )
        denum_ssh2u += torch.sum(  g_v * g_v )
        
        num_ssh2v += torch.sum( g_u * v_gt[:,:,1:-1,1:-1] )
        denum_ssh2v += torch.sum( g_u * g_u )
                
        #print(' Correlation dv_ssh/u : %f'% (torch.mean( g_v * u_gt[:,:,1:-1,1:-1] ) / torch.sqrt( torch.mean( g_v * g_v ) * torch.mean( u_gt[:,:,1:-1,1:-1] * u_gt[:,:,1:-1,1:-1] ) )))
        #print(' Correlation du_ssh/v : %f'% (torch.mean( g_u * v_gt[:,:,1:-1,1:-1] ) / torch.sqrt( torch.mean( g_u * g_u ) * torch.mean( v_gt[:,:,1:-1,1:-1] * v_gt[:,:,1:-1,1:-1] ) )))
        #print('%f %f '%(torch.mean( -1. * g_v * u_gt[:,:,1:-1,1:-1] )/torch.mean(  g_v * g_v ),torch.mean( g_u * v_gt[:,:,1:-1,1:-1] )/torch.mean( g_u * g_u )))
        #print( torch.mean( g_u * u_gt[:,:,1:-1,1:-1] ) / torch.sqrt( torch.mean( g_u * g_u ) * torch.mean( u_gt[:,:,1:-1,1:-1] * u_gt[:,:,1:-1,1:-1] ) ))
        #print( torch.mean( g_v * v_gt[:,:,1:-1,1:-1] ) / torch.sqrt( torch.mean( g_v * g_v ) * torch.mean( v_gt[:,:,1:-1,1:-1] * v_gt[:,:,1:-1,1:-1] ) ))

    alpha_ssh2u = num_ssh2u / denum_ssh2u
    alpha_ssh2v = num_ssh2v / denum_ssh2v
    
    alpha_ssh2u = alpha_ssh2u.detach().cpu().numpy()
    alpha_ssh2v = alpha_ssh2v.detach().cpu().numpy()

    print('.... SSH to (u,v) (normalized) : %f / %f '%(  alpha_ssh2u , alpha_ssh2v ))
    print('.... SSH to (u,v) : %f / %f '%( std_uv * alpha_ssh2u / stdTr , std_uv * alpha_ssh2v / stdTr ))

'''


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
         
        
'''

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
        
 '''       

       
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
        print(loss)
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
    
    #Différence de test en fonction de la reconstruction ou prévision :
    '''
    ##test_epoch_end reconstruction
    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds_debit'] for chunk in outputs]).numpy()
        x_test_rec = stdTrtrue * x_test_rec + meanTrtrue       
        self.x_rec_debit = x_test_rec[:,0,:,int(DT/2)]
        print(self.x_rec_debit.shape)
    '''    
    ##test epoch end prevision    
    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds_debit'] for chunk in outputs]).numpy()
        x_test_rec = stdTrtrue * x_test_rec + meanTrtrue
        self.x_rec_debit = x_test_rec[:,0,:,:]
        print(self.x_rec_debit.shape)
        #x_test_debit_obs = torch.cat([chunk['obs_debit'] for chunk in outputs]).numpy()
        #x_test_debit_obs[ x_test_debit_obs == 0. ] = np.float('NaN')
        #x_test_debit_obs = stdTr * x_test_debit_obs + meanTr
        #self.x_rec_debit_obs = x_test_debit_obs[:,:,:,int(DT/2)]

        return 1.

    def compute_loss(self, batch, phase):

        inputs_init,inputs_missing,masks,targets_GT = batch
        print(inputs_init.shape)
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
            
            #loss divergence KL:
            #X=(torch.moveaxis(outputs[:,0,:,int(2*DT/3+1):],0,1)).reshape(outputs.shape[2],outputs.shape[0]*(outputs.shape[3]-int(2*DT/3+1)))
            #Y=(torch.moveaxis(targets_GT[:,0,:,int(2*DT/3+1):],0,1)).reshape(outputs.shape[2],outputs.shape[0]*(outputs.shape[3]-int(2*DT/3+1)))
            #print(type(m_seuil[0]))
            #print(type(seuil_10[0]))
            #print("Nb>seuil")
            #print(torch.sum(X[0,:]>m_seuil[0,0]))
            #print(max(X[0,:]))
            #print("NbGT>seuil")
            #print(torch.sum(Y[0,:]>m_seuil[0,0]))
            #print(max(Y[0,:]))
                
            X = (torch.moveaxis(outputs[:,0,:,int(2*DT/3+1):],0,1)).reshape(outputs.shape[2],outputs.shape[0]*(outputs.shape[3]-int(2*DT/3+1)))
            
            Y=(torch.moveaxis(targets_GT[:,0,:,int(2*DT/3+1):],0,1)).reshape(outputs.shape[2],outputs.shape[0]*(outputs.shape[3]-int(2*DT/3+1)))
            Y1 = (torch.moveaxis(targets_GT[:,0,:,int(2*DT/3+1)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            X1 = (torch.moveaxis(outputs[:,0,:,int(2*DT/3+1)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            loss_KL1 = max(K_hat_tensor(m_seuil[:,0],X1,Y1)-K_hat_tensor(m_seuil[:,0],Y1,Y1),0)
            Y2 = (torch.moveaxis(targets_GT[:,0,:,int(2*DT/3+2)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            X2 = (torch.moveaxis(outputs[:,0,:,int(2*DT/3+2)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            loss_KL2 = max(K_hat_tensor(m_seuil[:,0],X2,Y2)-K_hat_tensor(m_seuil[:,0],Y2,Y2),0)
            Y3 = (torch.moveaxis(targets_GT[:,0,:,int(2*DT/3+3)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            X3 = (torch.moveaxis(outputs[:,0,:,int(2*DT/3+3)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            loss_KL3 = max(K_hat_tensor(m_seuil[:,0],X3,Y3)-K_hat_tensor(m_seuil[:,0],Y3,Y3),0)
            #loss_KL = max(K_hat_tensor(m_seuil[:,0],X,Y)-K_hat_tensor(m_seuil[:,0],Y,Y),0)
            loss_KL = loss_KL1+loss_KL2+loss_KL3
  
            #print("loss_KL")
            #print(0.1*loss_KL)
            #print("loss_All")
            #print(self.hparams.alpha[0] * loss_All)
            #print("loss_AE")
            #print(0.5 * self.hparams.alpha[1] * ( loss_AE + loss_AE_GT ))
            
            # total loss
            #loss        = self.hparams.alpha[0] * loss_All + 0.5 * self.hparams.alpha[1] * ( loss_AE + loss_AE_GT )+0.1*loss_KL
            #total loss without reanalysis
            loss = self.hparams.alpha[0] * loss_I + 0.5 * self.hparams.alpha[1] * ( loss_AE + loss_AE_GT )+0.1*loss_KL
            
            # metrics
            mse = loss_I.detach()
            metrics   = dict([('mse',mse)])
            #print(mse.cpu().detach().numpy())
            
            
            
            outputs = outputs
        return loss,outputs, metrics
    
    
'''
      
class LitModel(pl.LightningModule):
    def __init__(self,conf=HParam(),*args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # hyperparameters
        self.hparams.iter_update     = [0, 20, 40, 60, 100, 150, 800]  # [0,2,4,6,9,15]
        self.hparams.nb_grad_update  = [5, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.hparams.lr_update       = [1e-3, 1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        self.hparams.k_batch         = 1
        
        self.hparams.n_grad          = self.hparams.nb_grad_update[0]
        self.hparams.dim_grad_solver = dimGradSolver
        self.hparams.dropout         = rateDropout
        
        self.hparams.alpha_proj    = 0.5
        self.hparams.alpha_sr      = 0.5
        self.hparams.alpha_lr      = 0.5  # 1e4
        self.hparams.alpha_mse_ssh = 10.
        self.hparams.alpha_mse_gssh = 1.
        self.hparams.alpha_mse_vv = 1.
        
        self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(w_), requires_grad=False)
        self.hparams.automatic_optimization = False#True#

        self.hparams.alpha_ssh2u = alpha_ssh2u
        self.hparams.alpha_ssh2v = alpha_ssh2v
        self.hparams.d_du = 1.
        self.hparams.d_dv = alpha_ssh2u / alpha_ssh2v

        # main model
        self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                         Model_H(width_kernel=W_KERNEL_MODEL_H,dim=1), 
                                                         NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout), 
                                                         None, None, shapeData, self.hparams.n_grad)

        self.model_LR     = ModelLR()
        #self.gradient_img = Gradient_img()
        #self.compute_div = Div_uv()
        self.compute_graduv = Compute_graduv()
        self.w_loss       = self.hparams.w_loss # duplicate for automatic upload to gpu
        self.x_rec_ssh        = None # variable to store output of test method
        self.x_rec_u        = None # variable to store output of test method
        self.x_rec_v        = None # variable to store output of test method
        self.x_rec_ssh_obs = None
        self.x_rec_u_geo = None # variable to store output of test method
        self.x_rec_v_geo = None # variable to store output of test method
        self.x_feat_sst = None
        
        self.automatic_optimization = self.hparams.automatic_optimization
        self.curr = 0
        
    def forward(self):
        return 1

    def configure_optimizers(self):
        #optimizer = optim.Adam(self.model.parameters(), lr= self.lrUpdate[0])
        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                  {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.model_H.parameters(), 'lr': self.hparams.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5*self.hparams.lr_update[0]},
                                {'params': self.compute_graduv.parameters(), 'lr': self.hparams.lr_update[0]},
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
        self.log("tr_mse", metrics['mse_uv'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        self.log("val_uv", metrics['mse_uv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics = self.compute_loss(test_batch, phase='test')

        self.log('test_loss', loss)
        self.log("test_mse", metrics['mse'] / var_Tt , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_vv", metrics['mse_vv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        out_ssh,out_u_geo,out_v_geo,out_u,out_v, ssh_obs,sst_feat  = out
        return {'preds_ssh': out_ssh.detach().cpu(),'preds_u_geo': out_u_geo.detach().cpu(),'preds_v_geo': out_v_geo.detach().cpu(),'preds_u': out_u.detach().cpu(),'preds_v': out_v.detach().cpu(),'obs_ssh': ssh_obs.detach().cpu(),'feat_sst': sst_feat.detach().cpu()}

    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs
        print('.. \n')
    
    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds_ssh'] for chunk in outputs]).numpy()
        x_test_rec = stdTr * x_test_rec + meanTr
        
        self.x_rec_ssh = x_test_rec[:,int(dT/2),:,:]

        x_test_u = torch.cat([chunk['preds_u_geo'] for chunk in outputs]).numpy()
        x_test_u = std_uv * x_test_u
        self.x_rec_u_geo = x_test_u[:,int(dT/2),:,:]

        x_test_v = torch.cat([chunk['preds_v_geo'] for chunk in outputs]).numpy()
        x_test_v = std_uv * x_test_v
        self.x_rec_v_geo = x_test_v[:,int(dT/2),:,:]

        x_test_u = torch.cat([chunk['preds_u'] for chunk in outputs]).numpy()
        x_test_u = std_uv * x_test_u
        self.x_rec_u = x_test_u[:,int(dT/2),:,:]

        x_test_v = torch.cat([chunk['preds_v'] for chunk in outputs]).numpy()
        x_test_v = std_uv * x_test_v
        self.x_rec_v = x_test_v[:,int(dT/2),:,:]

        x_test_vv = torch.cat([chunk['preds_vv'] for chunk in outputs]).numpy()
        x_test_vv = std_vv * x_test_vv
        self.x_rec_vv = x_test_vv[:,int(dT/2),:,:]

        x_test_vv = torch.cat([chunk['preds_vv_swot'] for chunk in outputs]).numpy()
        x_test_vv = std_vv * x_test_vv
        self.x_rec_vv_swot = x_test_vv[:,int(dT/2),:,:]

        x_test_ssh_obs = torch.cat([chunk['obs_ssh'] for chunk in outputs]).numpy()
        x_test_ssh_obs[ x_test_ssh_obs == 0. ] = np.float('NaN')
        x_test_ssh_obs = stdTr * x_test_ssh_obs + meanTr
        self.x_rec_ssh_obs = x_test_ssh_obs[:,int(dT/2),:,:]

        x_test_sst_feat = torch.cat([chunk['feat_sst'] for chunk in outputs]).numpy()
        self.x_feat_sst = x_test_sst_feat

        return 1.

    def compute_loss(self, batch, phase):

        ssh_OI, imputs_obs, inputs_Mask, inputs_SST, ssh_GT, u_GT, v_GT, vv_GT = batch

        new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask, 0. * inputs_Mask , 0. * inputs_Mask ), dim=1)
        inputs_init    = torch.cat((ssh_OI, inputs_Mask * (ssh_GT - ssh_OI) , 0. * ssh_GT , 0. * ssh_GT ), dim=1)
        #inputs_missing = torch.cat((ssh_OI, inputs_Mask * (ssh_GT - ssh_OI) , 0. * ssh_GT , 0. * ssh_GT), dim=1)
        inputs_missing = torch.cat((ssh_OI, inputs_Mask * (imputs_obs - ssh_OI) , 0. * ssh_GT , 0. * ssh_GT), dim=1)
        
        mask_SST       = 1. + 0. * inputs_SST

        # gradient norm field
        g_targets_GT = self.gradient_img(ssh_GT)
        g_targets_GT = g_targets_GT[0]
        
       # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, [inputs_missing,inputs_SST], [new_masks,mask_SST])

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            outputsSLRHR = outputs
            outputs_ssh_lr = outputs[:, 0:dT, :, :]
            outputs_ssh = outputs[:, 0:dT, :, :] + outputs[:, dT:2*dT, :, :]
            outputs_uv1 = outputs[:, 2*dT:3*dT, :, :]  
            outputs_uv2 = outputs[:, 3*dT:4*dT, :, :]  

            # reconstruction losses for SSH
            g_outputs = self.gradient_img(outputs_ssh)
            g_outputs = g_outputs[0]
            loss_All   = NN_4DVar.compute_WeightedLoss((outputs_ssh - ssh_GT), self.w_loss)
            loss_GAll  = NN_4DVar.compute_WeightedLoss(g_outputs - g_targets_GT, self.w_loss)

            loss_OI    = NN_4DVar.compute_WeightedLoss(ssh_GT - ssh_OI, self.w_loss)
            
            loss_GOI   = NN_4DVar.compute_WeightedLoss(self.gradient_img(ssh_OI)[0] - g_targets_GT, self.w_loss)

            # reconstruction losses for current 
            if self.hparams.flag_uv_param =='div-curl':            
                curr_curl = self.model_pot2curr(outputs_ssh)
                curr_dcurl = self.model_pot2curr(outputs_uv1)
                curr_div = self.model_pot2curr(outputs_uv2)

                outputs_u = curr_curl[1] + curr_dcurl[1] + curr_div[0]
                outputs_v = -1. * curr_curl[0] - 1. * curr_dcurl[0] + curr_div[1]
    
                outputs_u_geo = curr_curl[1] + curr_dcurl[1]
                outputs_v_geo = -1. * curr_curl[0] - 1. * curr_dcurl[0] 

                loss_uv = NN_4DVar.compute_WeightedLoss((outputs_u - u_GT[:,:,1:-1,1:-1]), self.w_loss)
                loss_uv += NN_4DVar.compute_WeightedLoss((outputs_v - v_GT[:,:,1:-1,1:-1]), self.w_loss)
                
                loss_uv += NN_4DVar.compute_WeightedLoss((outputs_u_geo - u_GT[:,:,1:-1,1:-1]), self.w_loss)
                loss_uv += NN_4DVar.compute_WeightedLoss((outputs_v_geo - v_GT[:,:,1:-1,1:-1]), self.w_loss)
            else:
                outputs_u = outputs_uv1
                outputs_v = outputs_uv2

                curr_ssh = self.model_pot2curr(outputs_ssh)
                               
                outputs_u_geo = curr_ssh[1]
                outputs_v_geo = -1. * curr_ssh[0]              
                
                #loss_uv = NN_4DVar.compute_WeightedLoss((outputs_u - u_GT), self.w_loss)
                #loss_uv += NN_4DVar.compute_WeightedLoss((outputs_v - v_GT), self.w_loss)
                
                #loss_uv += NN_4DVar.compute_WeightedLoss((outputs_u_geo - u_GT[:,:,1:-1,1:-1]), self.w_loss)
                #loss_uv += NN_4DVar.compute_WeightedLoss((outputs_v_geo - v_GT[:,:,1:-1,1:-1]), self.w_loss)
                loss_uv = torch.sqrt( 1e-10 + (outputs_u - u_GT)**2 + (outputs_v - v_GT) **2 )
                loss_uv = compute_WeightedLoss_Lpqr(loss_uv, self.w_loss,self.hparams.p_norm_loss,self.hparams.q_norm_loss,self.hparams.r_norm_loss)
                
                loss_uv2 = torch.sqrt( 1e-10 + (outputs_u_geo - u_GT[:,:,1:-1,1:-1])**2 + (outputs_v_geo - v_GT[:,:,1:-1,1:-1]) **2 )
                loss_uv += compute_WeightedLoss_Lpqr(loss_uv2, self.w_loss,self.hparams.p_norm_loss,self.hparams.q_norm_loss,self.hparams.r_norm_loss)

                #loss_uv += NN_4DVar.compute_WeightedLoss((outputs_u_geo - u_GT[:,:,1:-1,1:-1]), self.w_loss)
                #loss_uv += NN_4DVar.compute_WeightedLoss((outputs_v_geo - v_GT[:,:,1:-1,1:-1]), self.w_loss)
                #div_rec = self.compute_div(outputs_u,outputs_v)
                #div_gt = self.compute_div(u_GT,v_GT)
                
                div_rec = model_div(outputs_u,outputs_v)
                div_gt = model_div(u_GT,v_GT)

                outputs_u = outputs_u[:,:,1:-1,1:-1]
                outputs_v = outputs_v[:,:,1:-1,1:-1]

                loss_div = NN_4DVar.compute_WeightedLoss((div_rec - div_gt), self.w_loss)
                #loss_uv += 10. * loss_div
                
                if self.current_epoch == 1:
                    self.curr = 1
                    
                if self.curr == self.current_epoch :
                    #print( np.mean( (div_gt.detach().cpu().numpy())**2 ), flush=True )
                    #print( np.mean( (div_rec.detach().cpu().numpy())**2 ), flush=True )

                    self.curr += 1

            # projection losses
            loss_AE     = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR) ** 2)
            yGT         = torch.cat((ssh_GT,ssh_GT-ssh_OI,outputs_uv1,outputs_uv2),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR      = NN_4DVar.compute_WeightedLoss(outputs_ssh_lr - ssh_OI, self.w_loss)
            targets_GTLR = self.model_LR(ssh_OI)
            loss_LR      = NN_4DVar.compute_WeightedLoss(self.model_LR(outputs_ssh) - targets_GTLR, self.w_loss)

            # total loss
            loss = self.hparams.alpha_mse_ssh * loss_All + self.hparams.alpha_mse_gssh * loss_GAll + self.hparams.alpha_mse_uv * loss_uv
            loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
            loss    += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
            
            # metrics
            mean_gall = NN_4DVar.compute_WeightedLoss(g_targets_GT,self.w_loss)
            mse = loss_All.detach()
            mse_grad   = loss_GAll.detach()
            mse_uv = loss_uv
            metrics   = dict([('mse',mse),('mse_grad',mse_grad),('mse_uv',mse_uv),('meanGrad',mean_gall),('mseOI',loss_OI.detach()),('mseGOI',loss_GOI.detach())])
            #print(mse.cpu().detach().numpy())
            
            outputs = [outputs_ssh,outputs_u_geo,outputs_v_geo,outputs_u,outputs_v,inputs_missing[:,dT:2*dT,:,:]]
        return loss,outputs, metrics
'''

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




'''
def compute_metrics(X_test,X_rec):
    # MSE
    mse = np.mean( (X_test - X_rec)**2 )

    # MSE for gradient
    gX_rec = np.gradient(X_rec,axis=[1,2])
    gX_rec = np.sqrt(gX_rec[0]**2 +  gX_rec[1]**2)
    
    gX_test = np.gradient(X_test,axis=[1,2])
    gX_test = np.sqrt(gX_test[0]**2 +  gX_test[1]**2)
    
    gmse = np.mean( (gX_test - gX_rec)**2 )
    ng   = np.mean( (gX_rec)**2 )
    
    return {'mse':mse,'mseGrad': gmse,'meanGrad': ng}

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
'''


if __name__ == '__main__':
    
    flagProcess = 3
    
    if flagProcess == 1: ## training model from scratch
    
        
        loadTrainedModel = 0#False#
        if loadTrainedModel == 1 :             
            
            pathCheckPOint = 'ResDanube4DVar/-exp3/model_Debit-epoch=08-val_loss=0.24.ckpt'
            
            print('.... load pre-trained model :'+pathCheckPOint)
            mod = LitModel.load_from_checkpoint(pathCheckPOint)
            #mod.compute_graduv = Compute_graduv()
 
            #mod.hparams.n_grad          = 5
            #mod.hparams.iter_update     = [0, 20, 40, 60, 150, 150, 800]  # [0,2,4,6,9,15]
            #mod.hparams.nb_grad_update  = [5, 5, 10, 10, 15, 20, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            #mod.hparams.lr_update       = [1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-7]
        else:
            mod = LitModel()
       
                   
        filename_chkpt = 'model_Debit_KL'
        
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

        
        pathCheckPOint = 'ResDanube4DVar/-exp3/model_Debit_KL-epoch=20-val_loss=0.29.ckpt'
        mod = LitModel.load_from_checkpoint(pathCheckPOint)     
        
        mod.hparams.n_grad = 5
        mod.hparams.median_filter_width = 1
        mod.hparams.flag_median_output = False
        width_med_filt_spatial = 1
        width_med_filt_temp = 1

        profiler_kwargs = {'max_epochs': 10}
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
        if flagTypeMissData == 4 : 
            X_val    =  X_valD[:,:,:]
            val_mseRec = compute_metrics(X_val,mod.x_rec_debit)            
            val_norm_rmse_debit = np.sqrt( np.mean( X_val.ravel()**2 ) )
            
        else : 
            X_val    =  dataValNoNaND[:,:,int(DT/2)]
        
        ## postprocessing
        #if width_med_filt_spatial + width_med_filt_temp > 2 :
            #mod.x_rec_debit = ndimage.median_filter(mod.x_rec_debit,size=(width_med_filt_temp,width_med_filt_spatial,width_med_filt_spatial))
        
            val_mseRec = compute_metrics(X_val,mod.x_rec_debit)     
       
            val_norm_rmse_debit = np.sqrt( np.mean( X_val.ravel()**2 ) )

        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        
        
        if flagTypeMissData==4 :
            X_Test   =  X_testD[:,:,:]
            #X_mask = []
        
            test_mseRec = compute_metrics(X_Test,mod.x_rec_debit)     
        
        
            test_norm_rmse_debit = np.sqrt( np.mean( X_Test.ravel()**2 ) )

            saveRes = True# 
            debit_gt = X_Test
        
            debit_obs = X_test_missingD[:,:,:]
        
            debit_rec = mod.x_rec_debit    
            
        else:     
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
                        debit_rec = debit_rec,
                        mode='prevision'
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
        
       