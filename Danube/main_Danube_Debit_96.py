#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:33:30 2020
@author: rfablet
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:09:32 2020
@author: rfablet
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt 
#import os
#import tensorflow.keras as keras

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

#############################################################################################
##
## Implementation of a NN pytorch framework for the identification and resolution 
## of 4DVar assimilation model applied to Lorenz-96 dynamics (40-dimensional state)
## 
## Data generation
##      flagProcess == 0: Simulation of L96 time series, inc. noise and sampling scheme of observed states
##      flagProcess == 1: Initial interpolation of the observed time series using a linear interpolation
##
## Bulding NN architectures
##      flagProcess == 2: Generation of the NN architecture for the dynamical prior given flagAEType variable
##        flagAEType == 0: CNN-based implementation of L96 ODE (RK4 or Euler schemes)
##        flagAEType == 1: single-scale GENN with DimAE-dimensional latent states
##        flagAEType == 2: two-scale GENN with DimAE-dimensional latent states
##
## Learning schemes
##      flagProcess == 3: Supervised training of the dynamical prior
##      flagProcess == 4: Joint supervised training of the dynamical prior and 4DVar solver 
##      (See torch_4DVarNN_L96 for the different types of 4DVar Solver)
##      flagProcess == 5: Joint training of the dynamical prior and 4DVar solver using test data (noisy/irregularly-sampled)
##      flagProcess == 6: Joint training of the dynamical prior and 4DVar solver using 4 + 5
##
## Interpolation of L96 data
##      flagProcess == 10: Apply a trained model for the evaluation of different performance metrics
##      flagProcess == 11: Iterative application of a trained model from linear interpolation initialization (cf above)
##      flagProcess == 12: Fixed-step gradient-based solver (autograd) of the 4DVAR Assimilation given the trained priod

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    flagProcess    = [0,1,2,3,4]#Sequence for processes to be run
    
    flagRandomSeed = 0
    flagSaveModel  = 1
     
  
    batch_size  = 96#4#4#8#12#8#256#8

    dirSAVE     = './ResDanube4DVar/'
    genFilename = 'Debit_v11'
  
    flagAEType = 2 # 0: L96 model, 1-2: GENN
    DimAE      = 50#50#10#50
    
    UsePriodicBoundary = True # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
    InterpFlag         = False

    NbDays          = 18244

    
    for kk in range(0,len(flagProcess)):
        
        ###############################################################
        ## data generation including noise sampling and missing data gaps
        if flagProcess[kk] == 0:        
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
    
             ####################################################
            ## Generation of training and test dataset
            ## Extraction of time series of dT time steps
            #NbTraining = 6000#2000
            #NbTest     = 256#256#500
            #NbVal = ?
            time_step  = 1
            DT = 21
            sigNoise   = np.sqrt(2)
            rateMissingData = 0.5#0.95
    
            dataTrainingNoNaND = image.extract_patches_2d(dataset[Indtrain[0][0]:Indtrain[0][1],:],(DT,31))         
            for k in Indtrain[1::]:
                d= image.extract_patches_2d(dataset[k[0]:k[1],:],(DT,31))
                dataTrainingNoNaND=np.concatenate((dataTrainingNoNaND,d),axis=0)
        
    
            dataValNoNaND = image.extract_patches_2d(dataset[Indval[0][0]:Indval[0][1],:],(DT,31))    
            for k in Indval[1::]:
                d= image.extract_patches_2d(dataset[k[0]:k[1],:],(DT,31))
                dataValNoNaND=np.concatenate((dataValNoNaND,d),axis=0)
             
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

                for i in range(31):
                    dataTrainingD[:,:,i] = dataTrainingNoNaND[:,:,i]
                    dataValD[:,:,i] = dataValNoNaND[:,:,i]
                    dataTestD[:,:,i] = dataTestNoNaND[:,:,i]
                for i in MaskedStations:
                    dataTrainingD[:,:,i-1] = float('nan')
                    dataValD[:,:,i-1] = float('nan')
                    dataTestD[:,:,i-1] = float('nan')
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
            
            flagTypeConstantMask=0
        
        
            
            
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
            
                mean3mask = np.mean(mean2mask,1)
                mean3mask = mean3mask.reshape(31,1)
            
                meanTr          = mean3/mean3mask
            
                mean2true = np.mean(X_trainD[:],0)
                mean3true = np.mean(mean2true,1)
                mean3true = mean3true.reshape(31,1)
                meanTrtrue = mean3true
            
                meansquaretrue = np.mean( (X_trainD-meanTrtrue)**2,0)
                meansquare2true = np.mean(meansquaretrue,1)
                meansquare2true=meansquare2true.reshape(31,1)
                stdTrtrue           = np.sqrt(meansquare2true )
            
                       
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

            
                x_trainD = (X_trainD - meanTrtrue) / stdTrtrue
                x_valD = (X_valD - meanTrtrue) / stdTrtrue
                x_testD  = (X_testD - meanTrtrue) / stdTrtrue
            
            
            
            else : 
                mean2 = np.mean(X_train_missingD[:],0)
                mean2mask = np.mean(mask_trainD[:],0)           
                mean3 = np.mean(mean2,1)
                mean3 = mean3.reshape(31,1)
            
                mean3mask = np.mean(mean2mask,1)
                mean3mask = mean3mask.reshape(31,1)
            
                meanTr          = mean3/mean3mask
                        
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
            
    
            
        ###############################################################
        ## Initial interpolation
        elif flagProcess[kk] == 1:        
            print('........ Initialize interpolated states')

            # Initialization for interpolation
            flagInit = 1
            
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
            
################################################
        ## AE architecture
        elif flagProcess[kk] == 2:        
            print('........ Define AE architecture')
            
            shapeData = np.ones(3).astype(int)
            shapeData[1:] =  x_trainD.shape[1:]
            # freeze all ode parameters
            
    
    
                    
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
 
            class Model_AE(torch.nn.Module):
                def __init__(self):
                    super(Model_AE, self).__init__()
                    self.encoder = Encoder()
                    self.decoder = Decoder()
            
                def forward(self, x):
                    x = self.encoder( x )
                    x = self.decoder( x )
                    return x

            model_AE           = Model_AE()
            print('AE Model type: '+genSuffixModel)
            print(model_AE)
            print('Number of trainable parameters = %d'%(sum(p.numel() for p in model_AE.parameters() if p.requires_grad)))

        ###############################################################
        ## Training AE from supervised data
        elif flagProcess[kk] == 3:        
            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_AE           = model_AE.to(device)
            
            # mean-squared error loss
            #criterion = torch.nn.MSELoss()
            var_Tr    = np.var( x_train )
            var_Tt    = np.var( x_test )

            #### Check AE performance 
            optimizer_AE       = optim.Adam(model_AE.parameters(), lr=1e-3)
            exp_lr_schedulerAE = lr_scheduler.StepLR(optimizer_AE, step_size=300, gamma=0.1)

            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_missing),torch.Tensor(x_train_obs),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_missing),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
            
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            }
            
            dataset_sizes = {'train': len(training_dataset), 'val': len(test_dataset)}

            # training function for dinAE
            since = time.time()
            
            best_model_AE_wts = copy.deepcopy(model_AE.state_dict())
            best_loss         = 1e10
            
            num_epochs = 1000
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
            
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model_AE.train()
                    else:
                        model_AE.eval()   # Set model to evaluate mode
            
                    running_loss = 0.0
                    num_loss     = 0
            
                    # Iterate over data.
                    #for inputs_ in dataloaders[phase]:
                    #    inputs = inputs_[0].to(device)
                    for inputs_init,inputs_missing,masks,targets_GT in dataloaders[phase]:
                        targets_GT     = targets_GT.to(device)
                        #print(inputs.size(0))
            
                        # zero the parameter gradients
                        optimizer_AE.zero_grad()
            
                        # reshaping tensors
                        targets_GT     = targets_GT.view(-1,1,targets_GT.size(1),targets_GT.size(2))

                        # forward
                        # need to evaluate grad/backward during the evaluation and training phase for model_AE
                        with torch.set_grad_enabled(True): 
                        #with torch.set_grad_enabled(phase == 'train'):
                            outputs = model_AE(targets_GT)
                            #outputs = model(inputs)
                            #loss = criterion( outputs,  inputs)
                            loss      = torch.mean((outputs - targets_GT)**2 )
            
                            # backward + optimize only if in training phase
                            if phase == 'train':
                              loss.backward()
                              optimizer_AE.step()
            
                        # statistics
                        running_loss             += loss.item() * inputs_missing.size(0)
                        num_loss                 += inputs_missing.size(0)
                        #running_expvar += torch.sum( (outputs - inputs)**2 ) / torch.sum(
                    if phase == 'train':
                        exp_lr_schedulerAE.step()
            
                    epoch_loss       = running_loss / num_loss
                    #epoch_acc = running_corrects.double() / dataset_sizes[phase]
                    if phase == 'train':
                        epoch_nloss = epoch_loss / var_Tr
                    else:
                        epoch_nloss = epoch_loss / var_Tt
            
            
                    #print('{} Loss: {:.4f} '.format(
                      #   phase, epoch_loss))
                    print('{} Loss: {:.4e} NLossAll: {:.4e} '.format(
                        phase, epoch_loss,epoch_nloss))
            
                    # deep copy the model
                    if phase == 'val' and epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model_AE.state_dict())
            
                print()
            
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val loss: {:4f}'.format(best_loss))

        ###############################################################
        ## Given a trained AE, train an assimilation solver
        elif flagProcess[kk] == 4:

            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(".... Device GPU: "+str(torch.cuda.is_available()))
            print(shapeData.shape)

            # mean-squared error loss
            #criterion = torch.nn.MSELoss()
            var_Tr    = np.var( x_train )
            var_Tt    = np.var( x_test )
                         
            alpha          = np.array([1.,0.1])
            alpha4DVar     = np.array([0.01,1.])#np.array([0.30,1.60])#

            flagLearnWithObsOnly = False #True # 
            lambda_LRAE          = 0.5 # 0.5

            GradType       = 1 # Gradient computation (0: subgradient, 1: true gradient/autograd)
            OptimType      = 2 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
            num_epochs     = 4000

            IterUpdate     = [0,30,100,150,200,250,400]#[0,2,4,6,9,15]
            NbProjection   = [0,0,0,0,0,0,0]#[0,0,0,0,0,0]#[5,5,5,5,5]##
            NbGradIter     = [0,5,10,15,20,20,20]#[0,0,1,2,3,3]#[0,2,2,4,5,5]#
            lrUpdate       = [1e-3,1e-4,1e-4,1e-4,1e-4,1e-5,1e-6,1e-6,1e-7]
            
            # NiterProjection,NiterGrad: global variables
            # bug for NiterProjection = 0
            #model_AE_GradFP = Model_AE_GradFP(model_AE2,shapeData,NiterProjection,NiterGrad,GradType,OptimType)
            NBGradCurrent   = NbGradIter[0]
            NBProjCurrent   = NbProjection[0]
            lrCurrent       = lrUpdate[0]
            
            model           = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
            modelSave       = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary) 
            model           = model.to(device)
            print('4DVar model: Number of trainable parameters = %d'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))
                         
            flagUseMultiGPU = False
            if flagUseMultiGPU == True :
                print('... Number of GPUs: %d'%torch.cuda.device_count())
                if torch.cuda.device_count() > 1:
                  print("Let's use", torch.cuda.device_count(), "GPUs!")
                  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                  modelMultiGPU = torch.nn.DataParallel(model)                
            
                  modelMultiGPU.to(device)

            flagLoadModel   = 0
            
            if flagLoadModel == 1:
                print('.... load model: '+fileAEModelInit)
                model.model_AE.load_state_dict(torch.load(fileAEModelInit))
                model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelAE_iter','_modelGrad_iter')))
            
            # optimization setting: freeze or not the AE
            optimizer   = optim.Adam([{'params': model.model_Grad.parameters()},
                                    {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                    ], lr=lrCurrent)
        
            #optimizer   = optim.RMSprop([{'params': model.model_Grad.parameters()},
            #                        {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
            #                        ], lr=lrCurrent)

            #for param in model.model_AE.parameters():
            #   param.requires_grad = False
            #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)            
 
            # Create training/test data pytorch tensors and associated  
            # list of tensors (xx[n][x] to access the nth sample for the xth field)
            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init),torch.Tensor(x_train_obs),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_Init),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
            
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            }            
            dataset_sizes = {'train': len(training_dataset), 'val': len(test_dataset)}
        
            # training function for dinAE
            since = time.time()
            
            if flagLearnWithObsOnly == True:
                model.model_Grad.compute_Grad.alphaObs = torch.nn.Parameter(torch.Tensor([np.sqrt(alpha4DVar[0])]).to(device))
                model.model_Grad.compute_Grad.alphaAE  = torch.nn.Parameter(torch.Tensor([np.sqrt(alpha4DVar[1])]).to(device))
            
                model.model_Grad.compute_Grad.alphaObs.requires_grad = False
                model.model_Grad.compute_Grad.alphaAE.requires_grad  = False
                
                genSuffixModelBase = genSuffixModel+'_WithObsOnly'
                alpha_Grad = alpha4DVar[0]
                #alpha_FP   = 1. - alpha[0]
                alpha_AE   = alpha4DVar[1]
                
                if lambda_LRAE > 0. :
                    print('... Fine-tuning of the prior model not consistent with flagLearnWithObsOnly  == True. Check')
                print('... alphaObs %.3f'%model.model_Grad.compute_Grad.alphaObs.item())    
                print('... alphaPrior %.3f'%model.model_Grad.compute_Grad.alphaAE.item())    
            else:
                genSuffixModelBase = genSuffixModel
                alpha_Grad = alpha[0]
                #alpha_FP   = 1. - alpha[0]
                alpha_AE   = alpha[1]
                
                # Suffix for file naming
                genSuffixModelBase = genSuffixModel
            
            genSuffixModel = genSuffixModelBase+'_DT'+str('%02d'%(time_step))+'_'+str('%03d'%(dT))
            genSuffixModel = genSuffixModel+genSuffixObs
            if lambda_LRAE == 0. :
                genSuffixModel = genSuffixModel+'_NoFTrAE'
            
            genSuffixModel = genSuffixModel+'_Nproj'+str('%02d'%(NBProjCurrent))
            genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(GradType))+'_'+str('%02d'%(OptimType))+'_'+str('%02d'%(NBGradCurrent))

            print('...... Suffix trained models: '+genSuffixModel)
    
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = 1e10
            
            comptUpdate = 1
            iterInit    = 0
            for epoch in range(iterInit,num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
            
                if ( epoch == IterUpdate[comptUpdate] ) & ( epoch > 0 ):
                    # update GradFP parameters
                    NBProjCurrent = NbProjection[comptUpdate]
                    NBGradCurrent = NbGradIter[comptUpdate]
                    lrCurrent     = lrUpdate[comptUpdate]
                    
                    if( (NBProjCurrent != NbProjection[comptUpdate-1]) | (NBGradCurrent != NbGradIter[comptUpdate-1]) ):
                        print("..... ")
                        print("..... ")
                        print("..... Update/initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection[comptUpdate],NbGradIter[comptUpdate]))
    
                        # update GradFP architectures
                        print('..... Update model architecture')
                        print("..... ")
                        model = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
                        model = model.to(device)
                        
                        if flagLearnWithObsOnly == True:
                            model.model_Grad.compute_Grad.alphaObs = torch.nn.Parameter(torch.Tensor([np.sqrt(alpha4DVar[0])]).to(device))
                            model.model_Grad.compute_Grad.alphaAE  = torch.nn.Parameter(torch.Tensor([np.sqrt(alpha4DVar[1])]).to(device))
                        
                            model.model_Grad.compute_Grad.alphaObs.requires_grad = False
                            model.model_Grad.compute_Grad.alphaAE.requires_grad  = False

                        # copy model parameters from current model
                        model.load_state_dict(best_model_wts)
                        
                        optimizer        = optim.Adam([{'params': model.model_Grad.parameters()},
                                                {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                                ], lr=lrCurrent)
    
                        # Suffix for file naming
                        genSuffixModel = genSuffixModelBase+'_DT'+str('%02d'%(time_step))+'_'+str('%03d'%(dT))
                        genSuffixModel = genSuffixModel+genSuffixObs
                        if lambda_LRAE == 0. :
                             genSuffixModel = genSuffixModel+'_NoFTrAE'
                        
                        genSuffixModel = genSuffixModel+'_Nproj'+str('%02d'%(NBProjCurrent))
                        genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(GradType))+'_'+str('%02d'%(OptimType))+'_'+str('%02d'%(NBGradCurrent))

                    else:
                        # update optimizer learning rate
                        print('..... Update learning rate')
                        mm = 0
                        lr = np.array([lrCurrent,lambda_LRAE*lrCurrent])
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr[mm]
                            mm += 1

                    # update counter
                    if comptUpdate < len(IterUpdate)-1:
                        comptUpdate += 1

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:        
                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()
            
                    running_loss         = 0.0
                    running_loss_All     = 0.
                    running_loss_R       = 0.
                    running_loss_I       = 0.
                    running_loss_AE      = 0.
                    num_loss             = 0
            
                    # Iterate over data.
                    #for inputs_ in dataloaders[phase]:
                    #    inputs = inputs_[0].to(device)
                    for inputs_init,inputs_missing,masks,targets_GT in dataloaders[phase]:
                        inputs_init    = inputs_init.to(device)
                        inputs_missing = inputs_missing.to(device)
                        masks          = masks.to(device)
                        targets_GT     = targets_GT.to(device)
                        #print(inputs.size(0))
            
            
                        # reshaping tensors
                        inputs_init    = inputs_init.view(-1,1,inputs_init.size(1),inputs_init.size(2))
                        inputs_missing = inputs_missing.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                        masks          = masks.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                        targets_GT     = targets_GT.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                                                
                        # zero the parameter gradients
                        optimizer.zero_grad()
            
                        # forward
                        # need to evaluate grad/backward during the evaluation and training phase for model_AE
                        with torch.set_grad_enabled(True): 
                        #with torch.set_grad_enabled(phase == 'train'):
                            inputs_init    = torch.autograd.Variable(inputs_init, requires_grad=True)
                            #if model.OptimType == 1:
                            #    outputs,grad_new,normgrad = model(inputs_init,inputs_missing,masks,None)
                            #    
                            #elif model.OptimType == 2:
                            #    outputs,hidden_new,cell_new,normgrad = model(inputs_init,inputs_missing,masks,None,None)
                                
                            #else:                               
                            #    outputs,normgrad = model(inputs_init,inputs_missing,masks)

                            if model.OptimType == 1:                                
                                if flagUseMultiGPU == True :
                                    outputs,grad_new,normgrad = modelMultiGPU(inputs_init,inputs_missing,masks,None)
                                else:
                                    outputs,grad_new,normgrad = model(inputs_init,inputs_missing,masks,None)
                                
                            elif model.OptimType == 2:
                                if flagUseMultiGPU == True :
                                    outputs,hidden_new,cell_new,normgrad = modelMultiGPU(inputs_init,inputs_missing,masks,None,None)
                                else:
                                    outputs,hidden_new,cell_new,normgrad = model(inputs_init,inputs_missing,masks,None,None)
                                
                            else:                               
                                if flagUseMultiGPU == True :
                                    outputs,normgrad = modelMultiGPU(inputs_init,inputs_missing,masks)
                                else:
                                    outputs,normgrad = model(inputs_init,inputs_missing,masks)
                              
                            loss_R      = torch.sum((outputs - targets_GT)**2 * masks )
                            loss_R      = torch.mul(1.0 / torch.sum(masks),loss_R)
                            loss_I      = torch.sum((outputs - targets_GT)**2 * (1. - masks) )
                            loss_I      = torch.mul(1.0 / torch.sum(1.-masks),loss_I)
                            loss_All    = torch.mean((outputs - targets_GT)**2 )
                            loss_AE     = torch.mean((model.model_AE(outputs) - outputs)**2 )
                                        
                            loss_AE_GT  = torch.mean((model.model_AE(targets_GT) - targets_GT)**2 )
                            #loss_AE_GT  = torch.mean((mod.model_AE(targets_GT) - targets_GT)**2 )
                                        #if phase == 'train':                                 
                            #    loss  = alpha_Grad * loss_All + 0.5 * alpha_AE * ( loss_AE + loss_AE_GT )
                            #else:
                            #    loss    = alpha_Grad * loss_R + alpha_AE * loss_AE
                            loss_Obs    = torch.sum( (outputs - inputs_missing)**2 * masks )
                            loss_Obs    = loss_Obs / torch.sum( masks )
            
                            if flagLearnWithObsOnly == True:
                                loss        = alpha4DVar[0] * loss_Obs + alpha4DVar[1] * loss_AE
                            else:
                                loss        = alpha_Grad * loss_All + 0.5 * alpha_AE * ( loss_AE + loss_AE_GT )
            
                            # backward + optimize only if in training phase
                            if( phase == 'train' ):
                                loss.backward()
                                optimizer.step()

                                #if flagLearnWithObsOnly == True:
                                #    print('.. Grad flag alphaObs: %s'%model.model_Grad.compute_Grad.alphaObs.requires_grad)
                                #    print('.. Current alpha values: %.3f  -- %.3f'%(model.model_Grad.compute_Grad.alphaObs.item(),model.model_Grad.compute_Grad.alphaAE.item()))
                                            
                        # statistics
                        running_loss             += loss.item() * inputs_missing.size(0)
                        running_loss_I           += loss_I.item() * inputs_missing.size(0)
                        running_loss_R           += loss_R.item() * inputs_missing.size(0)
                        running_loss_All         += loss_All.item() * inputs_missing.size(0)
                        running_loss_AE          += loss_AE_GT.item() * inputs_missing.size(0)
                        num_loss                 += inputs_missing.size(0)
            
                    epoch_loss       = running_loss / num_loss
                    epoch_loss_All   = running_loss_All / num_loss
                    epoch_loss_AE    = running_loss_AE / num_loss
                    epoch_loss_I     = running_loss_I / num_loss
                    epoch_loss_R     = running_loss_R / num_loss
                    #epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    epoch_loss     = epoch_loss * stdTr**2
                    epoch_loss_All = epoch_loss_All * stdTr**2
                    epoch_loss_I   = epoch_loss_I * stdTr**2
                    epoch_loss_R   = epoch_loss_R * stdTr**2
                    epoch_loss_AE  = epoch_loss_AE * stdTr**2

                    print('{} Loss: {:.4e} NLossAll: {:.4e} NLossR: {:.4e} NLossI: {:.4e} NLossAE: {:.4e}'.format(
                        phase, epoch_loss,epoch_loss_All,epoch_loss_R,epoch_loss_I,epoch_loss_AE),flush=True)
                    #print('... F %f'%model.model_AE.encoder.F)
                    
                    if( phase == 'val') and ( GradType == 2 ):
                        print('... alphaL1 = %.3f ----- alphaL2 = %.3f '%(model.model_Grad.compute_Grad.alphaL1,model.model_Grad.compute_Grad.alphaL2))

                    # deep copy the model
                    if phase == 'val' and epoch_loss < best_loss:
                        best_loss      = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
            
                # Save model
                if ( flagSaveModel == 1 )  & ( ( np.mod(epoch,25) == 0  ) | ( epoch == num_epochs - 1) )  :                          
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelAE_iter%03d'%(epoch)+'.mod'
                    modelSave.load_state_dict(best_model_wts)
                    print('.................. Auto-Encoder '+fileMod)
                    torch.save(modelSave.model_AE.state_dict(), fileMod)
    
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelGrad_iter%03d'%(epoch)+'.mod'
                    print('.................. Gradient model '+fileMod)
                    torch.save(modelSave.model_Grad.state_dict(), fileMod)

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val loss: {:4f}'.format(best_loss))

            # load best model weights
            model.load_state_dict(best_model_wts)
            
            # Apply current model to data
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }
            
            x_train_pred = []
            x_test_pred  = []
            for phase in ['train', 'val']:    
                model.eval()
            
                # Iterate over data.
                #for inputs_ in dataloaders[phase]:
                #    inputs = inputs_[0].to(device)
                for inputs_init,inputs_missing,masks,targets_GT in dataloaders[phase]:
                    inputs_init    = inputs_init.to(device)
                    inputs_missing = inputs_missing.to(device)
                    masks          = masks.to(device)
                    targets_GT     = targets_GT.to(device)
                        #print(inputs.size(0))

            # forward
            # need to evaluate grad/backward during the evaluation and training phase for model_AE
            with torch.set_grad_enabled(True): 
            #with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs_init,inputs_missing,masks)
     
            # store predictions
            if phase == 'train':
                if( len(x_train_pred) == 0 ):
                    x_train_pred = np.copy(outputs.cpu().detach().numpy())
                else:
                    x_train_pred = np.concatenate((x_train_pred,outputs.cpu().detach().numpy()),axis=0)
    
            if phase == 'val':
                if( len(x_test_pred) == 0 ):
                    x_test_pred = np.copy(outputs.cpu().detach().numpy())
                else:
                    x_test_pred = np.concatenate((x_test_pred,outputs.cpu().detach().numpy()),axis=0)
            

        ###############################################################################################
        ## Given a trained AE, train an assimilation solver using both groundtruthed data and test data 
        ## Possible update of the init state from a given epoch
        elif flagProcess[kk] == 6:

            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(".... Device GPU: "+str(torch.cuda.is_available()))
            #print(shapeData.shape)

            # mean-squared error loss
            #criterion = torch.nn.MSELoss()
            var_Tr    = np.var( x_train )
            var_Tt    = np.var( x_test )
                         
            alphaObs        = 1. * np.array([.1,1.])
            alphaGT         = 1. * np.array([1.,0.1])
            GradType        = 0 # Gradient computation (0: subgradient, 1: true gradient/autograd)
            OptimType       = 2 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
            iterStateUpdate = 100 
            num_epochs      = 5000                

            IterUpdate     = [0,200,600,1000,1200,1600]#[0,2,4,6,9,15]
            NbProjection   = [0,0,0,0,0,0,0]#[0,0,0,0,0,0]#[5,5,5,5,5]##
            NbGradIter     = [20,20,20,20,20,20]#,10[0,0,1,2,3,3]#[0,2,2,4,5,5]#
            lrUpdate       = [1e-4,1e-5,1e-4,1e-4,1e-4,1e-5,1e-6,1e-6,1e-7]
                
            NBGradCurrent   = NbGradIter[0]
            NBProjCurrent   = NbProjection[0]
            lrCurrent       = lrUpdate[0]
                
            #model           = Model_AEL96_GradNoisyData2(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType)        
            #model           = Model_AEL96_GradNoisyData2(model_AE,shapeData,NBGradCurrent,GradType,OptimType) 
            model           = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
            modelSave       = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary) 
            model           = model.to(device)
            print('.... Optim Type: %d'%model.OptimType)
             
            flagLoadModel   = 1
            fileAEModelInit = './ResL634DVar/l63_DinAE4DVar_L63EulerNN_DT01_200_ObsSub_87_20_Nproj01_Grad_01_02_20_modelAE_iter400.mod'
            fileAEModelInit = './ResL964DVar/l96_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_20_modelAE_iter400.mod'
            fileAEModelInit = './ResL964DVar/l96_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_01_20_modelAE_iter250.mod'
            
            fileAEModelInit = './ResL964DVar/l96_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_00_02_20_modelAE_iter800.mod'
            #fileAEModelInit = './ResL964DVar/l96_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_00_01_20_modelAE_iter800.mod'
            if flagLoadModel == 1:
                print('... Load trained model ' +  fileAEModelInit )
                model.model_AE.load_state_dict(torch.load(fileAEModelInit))
                model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelAE_iter','_modelGrad_iter')))
                    
            #for param in model.model_Grad.parameters():
            #   param.requires_grad = True
    
            lambda_LRAE = 0.5
            #optimizer_AE   = optim.Adam(model.model_AE.parameters(), lr=lambda_LRAE*lrCurrent)
            #optimizer_Grad = optim.Adam(model.model_Grad.parameters(), lr=lrCurrent)

            optimizer   = optim.Adam([{'params': model.model_Grad.parameters()},
                                    {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                    ], lr=lrCurrent)
    
            optimizer   = optim.RMSprop([{'params': model.model_Grad.parameters()},
                                    {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                    ], lr=lrCurrent)
            #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)            

            # file naming
            genSuffixModelBase = genSuffixModel+'Flag10_'
                
            genSuffixModel = genSuffixModelBase+'_DT'+str('%02d'%(time_step))+'_'+str('%03d'%(dT))
            genSuffixModel = genSuffixModel+genSuffixObs
            if lambda_LRAE == 0. :
                 genSuffixModel = genSuffixModel+'_NoFTrAE'
                
            genSuffixModel = genSuffixModel+'_Nproj'+str('%02d'%(NBProjCurrent))
            genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(GradType))+'_'+str('%02d'%(OptimType))+'_'+str('%02d'%(NBGradCurrent))

            #x_train_Curr = np.copy(x_train_Init)
            #x_test_Curr  = np.copy(x_test_Init)

            # Create training/test data pytorch tensors and associated  
            # list of tensors (xx[n][x] to access the nth sample for the xth field)
            
            x_Curr  = torch.Tensor( np.concatenate( (x_train_Init,x_test_Init) , axis=0 ))            
            x_obs   = torch.Tensor( np.concatenate( (x_train_obs,x_test_obs),axis=0) )
            x_mask  = torch.Tensor( np.concatenate( (mask_train,mask_test),axis=0) )
            x_GT    = torch.Tensor( np.concatenate( (x_train,x_test), axis=0 ) )

            if model.OptimType == 2:
                x_grad1 = torch.Tensor(np.random.randn(x_obs.size(0),model.model_Grad.DimState,x_obs.size(1),x_obs.size(2) ))
                x_grad2 = torch.Tensor(np.random.randn(x_obs.size(0),model.model_Grad.DimState,x_obs.size(1),x_obs.size(2)))
            else:
                x_grad1 = torch.Tensor(np.random.randn(x_obs.size(0),1,x_obs.size(1),x_obs.size(2)))
                x_grad2 = torch.Tensor(np.random.randn(x_obs.size(0),1,x_obs.size(1),x_obs.size(2)))
            
                        
            x_dataType = torch.Tensor(np.zeros((x_Curr.size(0),1)))
            x_dataType[0:x_train_Init.shape[0],0] = 1.
            x_idx      = torch.Tensor(np.arange(x_Curr.size(0)))

            x_dataset = torch.utils.data.TensorDataset(x_Curr,x_obs,x_mask,x_GT,x_dataType,x_idx,x_grad1,x_grad2) # create your datset
            
            dataloaders = {
                'dataset': torch.utils.data.DataLoader(x_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            }            
            dataset_sizes = {'dataset': len(x_dataset)}
        
            # Suffix for file naming
            # training function for dinAE
            since = time.time()
            
            alpha_Obs      = alphaObs[0]
            alpha_AE_Obs   = alphaObs[1]
            
            alpha_GT      = alphaGT[0]
            alpha_AE_GT   = alphaGT[1]
            
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = 1e10
            
            comptUpdate = 1
            iterInit    = 0
            sizeState   = torch.Tensor([x_Curr.size(1) * x_Curr.size(2)])
            sizeState   = sizeState.to(device)
            
            for epoch in range(iterInit,num_epochs):
                #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                #print('-' * 10)
            
                if ( model.OptimType >= 0 ) & ( epoch == IterUpdate[comptUpdate] ) & ( epoch > 0 ):
                    # update GradFP parameters
                    NBProjCurrent = NbProjection[comptUpdate]
                    NBGradCurrent = NbGradIter[comptUpdate]
                    lrCurrent     = lrUpdate[comptUpdate]
                    
                    print("..... ")
                    print("..... ")
                    print("..... Update/initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection[comptUpdate],NbGradIter[comptUpdate]))

                    # update GradFP architectures
                    if( (NBProjCurrent != NbProjection[comptUpdate-1]) | (NBGradCurrent != NbGradIter[comptUpdate-1]) ):
                        print('..... Update model architecture')
                        print("..... ")
                        #model           = NN_4DVar.Model_4DVarNN_Grad(model_AE,shapeData,NBGradCurrent,GradType,OptimType,UsePriodicBoundary)        
                        model           = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
                        model           = model.to(device)
                        
                        # copy model parameters from current model
                        model.load_state_dict(best_model_wts)
                    
                        # Suffix for file naming
                        genSuffixModel = genSuffixModelBase+'_DT'+str('%02d'%(time_step))+'_'+str('%03d'%(dT))
                        genSuffixModel = genSuffixModel+genSuffixObs
                        if lambda_LRAE == 0. :
                             genSuffixModel = genSuffixModel+'_NoFTrAE'
                        
                        genSuffixModel = genSuffixModel+'_Nproj'+str('%02d'%(NBProjCurrent))
                        genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(GradType))+'_'+str('%02d'%(OptimType))+'_'+str('%02d'%(NBGradCurrent))

                        optimizer   = optim.Adam([{'params': model.model_Grad.parameters()},
                                                {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                                ], lr=lrCurrent)
                
                        optimizer   = optim.RMSprop([{'params': model.model_Grad.parameters()},
                                                {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                                ], lr=lrCurrent)
                    else:
                        # update optimizer learning rate
                        print('..... Update learning rate')
                        mm = 0
                        lr = np.array([lrCurrent,lambda_LRAE*lrCurrent])
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr[mm]
                            mm += 1

                    # update counter
                    if comptUpdate < len(IterUpdate)-1:
                        comptUpdate += 1

                # Each epoch has a training and validation phase
                #for phase in ['val']:
                if 1*1:
                    model.train()
                    #if phase == 'train':
                    #    #rint('Learning')
                    #    model.train()  # Set model to training mode
                    #else:
                    #    #print('Evaluation')
                    #    model.eval()   # Set model to evaluate mode
            
                    running_loss             = 0.0
                    num_loss                 = 0
                    num_loss_Tt              = 0.
                    num_loss_Tr              = 0.
            
                    running_loss_All_Tr      = 0.0
                    running_loss_I_Tr        = 0.0
                    running_loss_R_Tr        = 0.0
                    running_loss_AE_Tr       = 0.0
                    running_loss_AE_GT_Tr    = 0.0

                    running_loss_All_Tt      = 0.0
                    running_loss_I_Tt        = 0.0
                    running_loss_R_Tt        = 0.0
                    running_loss_AE_Tt       = 0.0
                    running_loss_AE_GT_Tt    = 0.0

                    # Iterate over data.
                    #for inputs_ in dataloaders[phase]:
                    #    inputs = inputs_[0].to(device)
                    for inputs_init,inputs_missing,masks,targets_GT,dataType,idx,grad1,grad2 in dataloaders['dataset']:
                        inputs_init    = inputs_init.to(device)
                        inputs_missing = inputs_missing.to(device)
                        masks          = masks.to(device)
                        targets_GT     = targets_GT.to(device)
                        dataType       = dataType.to(device)
                        #print(inputs.size(0))
                                                
                        # reshaping tensors
                        inputs_init    = inputs_init.view(-1,1,inputs_init.size(1),inputs_init.size(2))
                        inputs_missing = inputs_missing.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                        masks          = masks.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                        targets_GT     = targets_GT.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                                                
                        if model.OptimType == 1:
                            grad_old = grad1.to(device)
                        if model.OptimType == 2:
                            hidden = grad1.to(device)
                            cell   = grad2.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()
            
                        # forward
                        # need to evaluate grad/backward during the evaluation and training phase for model_AE
                        with torch.set_grad_enabled(True): 
                        #with torch.set_grad_enabled(phase == 'train'):
                            inputs_init    = torch.autograd.Variable(inputs_init, requires_grad=True)
                            
                            if model.OptimType == 1:
                                grad_old          = torch.autograd.Variable(grad_old, requires_grad=True)
                                #outputs,grad_new = model(inputs_init,inputs_missing,masks,grad_old)
                                outputs,grad_new = model(inputs_init,inputs_missing,masks,grad_old)
                                
                            elif model.OptimType == 2:
                                hidden          = torch.autograd.Variable(hidden, requires_grad=True)
                                cell            = torch.autograd.Variable(cell, requires_grad=True)
                                #outputs,hidden_new,cell_new = model(inputs_init,inputs_missing,masks,hidden,cell)
                                outputs,hidden_new,cell_new = model(inputs_init,inputs_missing,masks,None,None)
                                
                            else:                               
                                outputs = model(inputs_init,inputs_missing,masks)
                           
                            # minimized loss types
                            #dB           = 0
                            #dT           = outputs.size(3)
                            #loss_R       = torch.sum( (outputs[:,:,:,dB:dT-dB] - inputs_missing[:,:,:,dB:dT-dB])**2 * masks[:,:,:,dB:dT-dB] )

                            #loss_R_GT    = torch.sum( torch.sum( (outputs[:,:,:,dB:dT-dB] - targets_GT[:,:,:,dB:dT-dB])**2 , dim = 3) , dim = 2 )
                            #loss_R_GT    = torch.sum( dataType * loss_R_GT )

                            #loss_All     = torch.mean( (outputs[:,:,:,dB:dT-dB] - targets_GT[:,:,:,dB:dT-dB])**2 )
                            
                            #loss_AE      = torch.mean( (model.model_AE(outputs)[:,:,:,dB:dT-dB] - outputs[:,:,:,dB:dT-dB])**2 )                            
                            #loss_AE_GT   = torch.sum( torch.sum( (model.model_AE(targets_GT)[:,:,:,dB:dT-dB] - targets_GT[:,:,:,dB:dT-dB])**2 , dim = 3) , dim = 2 )
                            #loss_AE_GT   = torch.sum( dataType * loss_AE_GT )
                  

                            #loss_AE_GT  = torch.mean( (model.model_AE(targets_GT)[:,:,:,dB:dT-dB] - targets_GT[:,:,:,dB:dT-dB])**2 )
                            
                            # loss dataType == 1 (data with known groudn-truth)
                            # losses for training data= dataType == 1
                            eps = 0.001
                            loss_All_Tr    = torch.sum( torch.sum( (outputs - targets_GT)**2 , dim = 3) , dim = 2 ) / sizeState
                            loss_All_Tr    = torch.sum( dataType * loss_All_Tr ) / ( torch.sum( dataType ) + eps )
                            #loss_All_Tr    = torch.mean( (outputs - targets_GT)**2 )

                            loss_I_Tr      = torch.sum( torch.sum( (outputs - targets_GT)**2 * (1. - masks) , dim = 3) , dim = 2 )
                            loss_I_Tr      = torch.sum( dataType * loss_I_Tr ) 
                            loss_I_Tr      = loss_I_Tr / ( torch.sum( dataType * torch.sum( torch.sum(1. - masks, dim = 3) , dim = 2) ) + eps )

                            loss_R_Tr      = torch.sum( torch.sum( (outputs - targets_GT)**2 * masks , dim = 3) , dim = 2 )
                            loss_R_Tr      = torch.sum( dataType * loss_R_Tr ) 
                            loss_R_Tr      = loss_R_Tr / ( torch.sum( dataType * torch.sum( torch.sum(masks, dim = 3) , dim = 2) ) + eps )

                            loss_AE_Tr     = torch.sum( torch.sum( (outputs - model.model_AE(outputs))**2  , dim = 3) , dim = 2 ) / sizeState
                            loss_AE_Tr     = torch.sum( dataType * loss_AE_Tr ) / ( torch.sum( dataType ) + eps )

                            loss_AE_GT_Tr     = torch.sum( torch.sum( (targets_GT - model.model_AE(targets_GT))**2  , dim = 3) , dim = 2 ) / sizeState
                            loss_AE_GT_Tr     = torch.sum( dataType * loss_AE_GT_Tr ) / ( torch.sum( dataType ) + eps )
                           
                            loss_All_Tt    = torch.sum( torch.sum( (outputs - targets_GT)**2 , dim = 3) , dim = 2 ) / sizeState
                            loss_All_Tt    = torch.sum( (1.-dataType) * loss_All_Tt ) / ( torch.sum( 1. - dataType ) + eps )

                            loss_I_Tt      = torch.sum( torch.sum( (outputs - targets_GT)**2 * (1. - masks) , dim = 3) , dim = 2 )
                            loss_I_Tt      = torch.sum( (1.-dataType) * loss_I_Tt ) 
                            loss_I_Tt      = loss_I_Tt / ( torch.sum( (1.-dataType) * torch.sum( torch.sum(1. - masks, dim = 3) , dim = 2) ) + eps )

                            loss_R_Tt      = torch.sum( torch.sum( (outputs - targets_GT)**2 * masks , dim = 3) , dim = 2 )
                            loss_R_Tt      = torch.sum( (1.-dataType) * loss_R_Tt ) 
                            loss_R_Tt      = loss_R_Tt / ( torch.sum( (1.-dataType) * torch.sum( torch.sum(masks, dim = 3) , dim = 2) ) + eps )

                            loss_AE_Tt     = torch.sum( torch.sum( (outputs - model.model_AE(outputs))**2  , dim = 3) , dim = 2 ) / sizeState
                            loss_AE_Tt     = torch.sum( (1.-dataType) * loss_AE_Tt ) / ( torch.sum( (1.-dataType) ) + eps )

                            loss_AE_GT_Tt     = torch.sum( torch.sum( (targets_GT - model.model_AE(targets_GT))**2  , dim = 3) , dim = 2 ) / sizeState
                            loss_AE_GT_Tt     = torch.sum( (1.-dataType) * loss_AE_GT_Tt ) / ( torch.sum( (1.-dataType) ) + eps )

                            # loss to be minimized
                            nbTr        = ( torch.sum( dataType ) + eps )
                            nbTt        = ( torch.sum( 1.0 - dataType ) + eps )
                            loss        = nbTt * ( alpha_Obs * loss_R_Tt + alpha_AE_Obs * loss_AE_Tt )
                            loss       += nbTr * ( alpha_GT  * loss_All_Tr +  alpha_AE_GT * loss_AE_GT_Tr )
                            loss       += nbTr * ( alpha_Obs * loss_R_Tr + alpha_AE_Obs * loss_AE_Tr )

                            loss.backward(retain_graph=True)
                            
                            if model.OptimType >= 0 :
                                optimizer.step()
                                #print(model.model_Grad.gradNet1.weight)
                                #grad    = torch.autograd.grad(loss,x,create_graph=True)[0]
                                #grad.retain_grad()
                                #x     = x - delta * grad
                               
                                # copy the ouput in the input tensor for batch 
                                # after a few iterations
                                if( ( epoch >= iterStateUpdate ) & ( np.mod(epoch,10) == 0 ) ):
                                    print('.... Update init state')
                                    idx_batch = idx.detach().numpy().astype(int)
                                      
                                    UpdateInit = 1
                                    if UpdateInit == 0:
                                        delta = 1.0
                                        x     = inputs_init - delta * inputs_init.grad.data
                                        x_Curr[idx_batch,:,:] = x.detach().clone().view(-1,outputs.size(2),outputs.size(3)).cpu()
                                        
                                        if model.OptimType == 1:
                                            grad_old          = grad_old - delta * grad_old.grad.data                                    
                                            x_grad1[idx_batch,:,:,:] = grad_old.detach().clone().cpu()
                                        elif model.OptimType == 2:
                                            hidden            = hidden - delta * hidden.grad.data
                                            cell              = cell - delta * cell.grad.data
                                            
                                            x_grad1[idx_batch,:,:] = hidden.detach().clone().cpu()
                                            x_grad2[idx_batch,:,:] = cell.detach().clone().cpu()
                                    elif UpdateInit == 1:                                   
                                        beta = 1.0
                                        x     = (1.-beta) * inputs_init + beta * outputs
                                        x_Curr[idx_batch,:,:] = x.detach().clone().view(-1,outputs.size(2),outputs.size(3)).cpu()
                                        
                                        if model.OptimType == 1:
                                            grad_old          = (1.-beta) * grad_old + beta * grad_new                                   
                                            x_grad1[idx_batch,:,:,:] = grad_old.detach().clone().cpu()
                                        elif model.OptimType == 2:
                                            hidden            = (1.-beta) * hidden + beta * hidden_new                                   
                                            cell              = (1.-beta) * hidden + beta * cell_new                                   
                                            
                                            x_grad1[idx_batch,:,:] = hidden.detach().clone().cpu()
                                            x_grad2[idx_batch,:,:] = cell.detach().clone().cpu()
                            else :
                                idx_batch = idx.detach().numpy().astype(int)
                                x     = inputs_init -  delta * inputs_init.grad.data 
                                x_Curr[idx_batch,:,:] = x.detach().clone().view(-1,outputs.size(2),outputs.size(3)).cpu()
                        
                        running_loss             += loss.item()

                        running_loss_All_Tr      += loss_All_Tr.item() * nbTr
                        running_loss_I_Tr        += loss_I_Tr.item() * nbTr
                        running_loss_R_Tr        += loss_R_Tr.item() * nbTr
                        running_loss_AE_Tr       += loss_AE_Tr.item() * nbTr
                        running_loss_AE_GT_Tr    += loss_AE_GT_Tr.item() * nbTr

                        running_loss_All_Tt      += loss_All_Tt.item() * nbTt
                        running_loss_I_Tt        += loss_I_Tt.item() * nbTt
                        running_loss_R_Tt        += loss_R_Tt.item() * nbTt
                        running_loss_AE_Tt       += loss_AE_Tt.item() * nbTt
                        running_loss_AE_GT_Tt    += loss_AE_GT_Tt.item() * nbTt

                        num_loss                  = inputs_missing.size(0)
                        num_loss_Tr              += nbTr
                        num_loss_Tt              += nbTt
                        #running_expvar += torch.sum( (outputs - inputs)**2 ) / torch.sum(
                    #if phase == 'train':
                    #    exp_lr_scheduler.step()
            
                    epoch_loss           = running_loss / (num_loss)
                    
                    epoch_loss_All_Tr    = running_loss_All_Tr * stdTr**2 / num_loss_Tr
                    epoch_loss_AE_Tr     = running_loss_AE_Tr * stdTr**2 / num_loss_Tr
                    epoch_loss_AE_GT_Tr  = running_loss_AE_GT_Tr * stdTr**2 / num_loss_Tr
                    epoch_loss_I_Tr      = running_loss_I_Tr * stdTr**2 / num_loss_Tr                                        
                    epoch_loss_R_Tr      = running_loss_R_Tr * stdTr**2 / num_loss_Tr
                    
                    epoch_loss_All_Tt    = running_loss_All_Tt * stdTr**2 / num_loss_Tt
                    epoch_loss_AE_Tt     = running_loss_AE_Tt * stdTr**2 / num_loss_Tt
                    epoch_loss_AE_GT_Tt  = running_loss_AE_GT_Tt * stdTr**2 / num_loss_Tt
                    epoch_loss_I_Tt      = running_loss_I_Tt * stdTr**2 / num_loss_Tt                                        
                    epoch_loss_R_Tt      = running_loss_R_Tt * stdTr**2 / num_loss_Tt

                    #print('.... Total loss %.4e'%epoch_loss,flush=True)
                    #print('.... [Obs data] NLossAll: %.4e NLossR: %.4e NLossI: %.4e NLossAE: %.4e NLossAEGT: %.4e'%(epoch_loss_All,epoch_loss_R,epoch_loss_I,epoch_loss_AE,epoch_loss_AE_GT))
                    #print()
                    print('.. epoch %d/%d --- Total Loss: %.4e'%(epoch,num_epochs,epoch_loss))
                    print('.. Training NLossAll: %.4e NLossR: %.4e NLossI: %.4e NLossAE: %.4e NLossAEGT: %.4e'%(epoch_loss_All_Tr,epoch_loss_R_Tr,epoch_loss_I_Tr,epoch_loss_AE_Tr,epoch_loss_AE_GT_Tr))
                    print('.. Test     NLossAll: %.4e NLossR: %.4e NLossI: %.4e NLossAE: %.4e NLossAEGT: %.4e'%(epoch_loss_All_Tt,epoch_loss_R_Tt,epoch_loss_I_Tt,epoch_loss_AE_Tt,epoch_loss_AE_GT_Tt))
           
                    # update dataset in dataloader
                    x_dataset = torch.utils.data.TensorDataset(x_Curr,x_obs,x_mask,x_GT,x_dataType,x_idx,x_grad1,x_grad2) # create your datset
                    dataloaders = {
                            'dataset': torch.utils.data.DataLoader(x_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                            }            
                    # deep copy the model
                    if epoch_loss < best_loss:
                        best_loss      = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())

                # Save model
                if ( flagSaveModel == 1 )  & ( model.OptimType >= 0 ) & ( ( np.mod(epoch,10) == 0  ) | ( epoch == num_epochs - 1) )  :                          
                    # load best model weights
                    modelSave.load_state_dict(best_model_wts)

                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelAE_iter%03d'%(epoch)+'.mod'
                    print('.................. Auto-Encoder '+fileMod)
                    torch.save(modelSave.model_AE.state_dict(), fileMod)
    
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelGrad_iter%03d'%(epoch)+'.mod'
                    print('.................. Gradient model '+fileMod)
                    torch.save(modelSave.model_Grad.state_dict(), fileMod)
                    
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        
        ########################################################################
        ## Given a trained AE, train an assimilation solver using only test data 
        ## (no groundtruthed gap-free/noise-free data)
        ## iterative update of the state
        elif flagProcess[kk] == 5:

            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(".... Device GPU: "+str(torch.cuda.is_available()))
            #print(shapeData.shape)

            # mean-squared error loss
            #criterion = torch.nn.MSELoss()
            var_Tr    = np.var( x_train )
            var_Tt    = np.var( x_test )
                         
            alpha           = 1e0* np.array([.1,1.])
            GradType        = 1 # Gradient computation (0: subgradient, 1: true gradient/autograd)
            OptimType       = 1 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
            iterStateUpdate = 250 
            num_epochs      = 5000                

            IterUpdate     = [0,200,600,1000,1200,1600]#[0,2,4,6,9,15]
            #NbProjection   = [0,0,0,0,0,0,0]#[0,0,0,0,0,0]#[5,5,5,5,5]##
            NbGradIter     = [5,5,5,5,5,5]#,10[0,0,1,2,3,3]#[0,2,2,4,5,5]#
            lrUpdate       = [1e-3,1e-4,1e-3,1e-4,1e-4,1e-5,1e-6,1e-6,1e-7]

                
            
            # NiterProjection,NiterGrad: global variables
            # bug for NiterProjection = 0
            #model_AE_GradFP = Model_AE_GradFP(model_AE2,shapeData,NiterProjection,NiterGrad,GradType,OptimType)
            NBGradCurrent   = NbGradIter[0]
            NBProjCurrent   = NbProjection[0]
            lrCurrent       = lrUpdate[0]

            if OptimType == -1:
                NBGradCurrent = 0
                NBProjCurrent = 0
                iterStateUpdate = 0
                
            #model           = Model_AEL96_GradNoisyData2(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType)        
            if OptimType >= 0:                
                model           = NN_4DVar.Model_4DVarNN_Grad(model_AE,shapeData,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary) 
            else:
                model           = NN_4DVar.Model_4DVarNN_Grad(model_AE,shapeData,NBGradCurrent,GradType,0,InterpFlag,UsePriodicBoundary) 
                model.OptimType = -1
            model           = model.to(device)
            print('.... Optim Type: %d'%model.OptimType)
            print('4DVar model: Number of trainable parameters = %d'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))
             
            flagLoadModel   = 0
            fileAEModelInit = './ResL634DVar/l63_DinAE4DVar_L63EulerNN_DT01_200_ObsSub_87_20_Nproj01_Grad_01_02_20_modelAE_iter400.mod'
            if flagLoadModel == 1:
                model.model_AE.load_state_dict(torch.load(fileAEModelInit))
                model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelAE_iter','_modelGrad_iter')))
            genSuffixModelBase = genSuffixModel+'Flag9_'
                    
            #for param in model.model_Grad.parameters():
            #   param.requires_grad = True
    
            lambda_LRAE = 0.0
            #optimizer_AE   = optim.Adam(model.model_AE.parameters(), lr=lambda_LRAE*lrCurrent)
            #optimizer_Grad = optim.Adam(model.model_Grad.parameters(), lr=lrCurrent)

            optimizer   = optim.Adam([{'params': model.model_Grad.parameters()},
                                    {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                    ], lr=lrCurrent)
    
            optimizer   = optim.RMSprop([{'params': model.model_Grad.parameters()},
                                    {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                    ], lr=lrCurrent)
            #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)            

            #x_train_Curr = np.copy(x_train_Init)
            #x_test_Curr  = np.copy(x_test_Init)

            # Create training/test data pytorch tensors and associated  
            # list of tensors (xx[n][x] to access the nth sample for the xth field)
            
            x_Init  = x_test_Init
            x_Curr  = torch.Tensor(np.copy( x_Init ))
            #x_Curr  = torch.Tensor(np.random.randn(x_Init.shape[0],x_Init.shape[1],x_Init.shape[2]))
            
            x_obs   = torch.Tensor(x_test_obs)
            x_mask  = torch.Tensor(mask_test)
            x_GT    = torch.Tensor(x_test)
            if model.OptimType == 2:
                x_grad1 = torch.Tensor(np.random.randn(x_test.shape[0],model.model_Grad.DimState,x_test.shape[1],x_test.shape[2]))
                x_grad2 = torch.Tensor(np.random.randn(x_test.shape[0],model.model_Grad.DimState,x_test.shape[1],x_test.shape[2]))
            else:
                x_grad1 = torch.Tensor(np.random.randn(x_test.shape[0],1,x_test.shape[1],x_test.shape[2]))
                x_grad2 = torch.Tensor(np.random.randn(x_test.shape[0],1,x_test.shape[1],x_test.shape[2]))
            
                        
            x_dataType = torch.Tensor(np.zeros((x_Curr.shape[0],1)))
            x_idx      = torch.Tensor(np.arange(x_Curr.shape[0]))

            x_dataset = torch.utils.data.TensorDataset(x_Curr,x_obs,x_mask,x_GT,x_dataType,x_idx,x_grad1,x_grad2) # create your datset
            
            #training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Curr),torch.Tensor(x_train_obs),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            #test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_Curr),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
            
            dataloaders = {
                'dataset': torch.utils.data.DataLoader(x_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            }            
            dataset_sizes = {'dataset': len(x_dataset)}
        
            # Suffix for file naming
            # training function for dinAE
            since = time.time()
            
            alpha_Grad = alpha[0]
            #alpha_FP   = 1. - alpha[0]
            alpha_AE   = alpha[1]
            
            model.model_Grad.compute_Grad.alphaObs = torch.nn.Parameter(torch.Tensor([np.sqrt(alpha_Grad)]).to(device))
            model.model_Grad.compute_Grad.alphaAE  = torch.nn.Parameter(torch.Tensor([np.sqrt(alpha_AE)]).to(device))
            
            model.model_Grad.compute_Grad.alphaObs.requires_grad = False
            model.model_Grad.compute_Grad.alphaAE.requires_grad  = False
            
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = 1e10
            
            delta = torch.nn.Parameter(torch.Tensor([5.0e4]))
            delta = delta.to(device)
            
            comptUpdate = 1
            iterInit    = 0
            for epoch in range(iterInit,num_epochs):
                #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                #print('-' * 10)
            
                if ( model.OptimType >= 0 ) & ( epoch == IterUpdate[comptUpdate] ) & ( epoch > 0 ):
                    # update GradFP parameters
                    NBProjCurrent = NbProjection[comptUpdate]
                    NBGradCurrent = NbGradIter[comptUpdate]
                    lrCurrent     = lrUpdate[comptUpdate]
                    
                    print("..... ")
                    print("..... ")
                    print("..... Update/initialize number of projections/Graditer in GradCOnvAE model # %d/%d"%(NbProjection[comptUpdate],NbGradIter[comptUpdate]))

                    # update GradFP architectures
                    if( (NBProjCurrent != NbProjection[comptUpdate-1]) | (NBGradCurrent != NbGradIter[comptUpdate-1]) ):
                        print('..... Update model architecture')
                        print("..... ")
                        model           = NN_4DVar.Model_4DVarNN_Grad(model_AE,shapeData,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
                        model           = model.to(device)
                        
                        # copy model parameters from current model
                        model.load_state_dict(best_model_wts)
                    
                        # Suffix for file naming
                        genSuffixModel = genSuffixModelBase+'_DT'+str('%02d'%(time_step))+'_'+str('%03d'%(dT))
                        genSuffixModel = genSuffixModel+genSuffixObs
                        if lambda_LRAE == 0. :
                             genSuffixModel = genSuffixModel+'_NoFTrAE'
                        
                        genSuffixModel = genSuffixModel+'_Nproj'+str('%02d'%(NBProjCurrent))
                        genSuffixModel = genSuffixModel+'_Grad_'+str('%02d'%(GradType))+'_'+str('%02d'%(OptimType))+'_'+str('%02d'%(NBGradCurrent))

                        optimizer   = optim.Adam([{'params': model.model_Grad.parameters()},
                                                {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                                ], lr=lrCurrent)
                
                        optimizer   = optim.RMSprop([{'params': model.model_Grad.parameters()},
                                                {'params': model.model_AE.encoder.parameters(), 'lr': lambda_LRAE*lrCurrent}
                                                ], lr=lrCurrent)
                    else:
                        # update optimizer learning rate
                        print('..... Update learning rate')
                        mm = 0
                        lr = np.array([lrCurrent,lambda_LRAE*lrCurrent])
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr[mm]
                            mm += 1

                    # update counter
                    if comptUpdate < len(IterUpdate)-1:
                        comptUpdate += 1

                # Each epoch has a training and validation phase
                #for phase in ['val']:
                if 1*1:
                    model.train()
                    #if phase == 'train':
                    #    #rint('Learning')
                    #    model.train()  # Set model to training mode
                    #else:
                    #    #print('Evaluation')
                    #    model.eval()   # Set model to evaluate mode
            
                    running_loss         = 0.0
                    running_loss_All     = 0.
                    running_loss_R       = 0.
                    running_loss_I       = 0.

                    running_loss_AE      = 0.
                    running_loss_AE_GT   = 0.
                    num_loss             = 0
            
                    # Iterate over data.
                    #for inputs_ in dataloaders[phase]:
                    #    inputs = inputs_[0].to(device)
                    for inputs_init,inputs_missing,masks,targets_GT,dataType,idx,grad1,grad2 in dataloaders['dataset']:
                        inputs_init    = inputs_init.to(device)
                        inputs_missing = inputs_missing.to(device)
                        masks          = masks.to(device)
                        targets_GT     = targets_GT.to(device)
                        dataType       = dataType.to(device)
                        #print(inputs.size(0))
                                                
                        # reshaping tensors
                        inputs_init    = inputs_init.view(-1,1,inputs_init.size(1),inputs_init.size(2))
                        inputs_missing = inputs_missing.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                        masks          = masks.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                        targets_GT     = targets_GT.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                                                
                        if model.OptimType == 1:
                            grad_old = grad1.to(device)
                        if model.OptimType == 2:
                            hidden = grad1.to(device)
                            cell   = grad2.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()
            
                        # forward
                        # need to evaluate grad/backward during the evaluation and training phase for model_AE
                        with torch.set_grad_enabled(True): 
                        #with torch.set_grad_enabled(phase == 'train'):
                            inputs_init    = torch.autograd.Variable(inputs_init, requires_grad=True)
                            
                            if model.OptimType == 1:
                                grad_old          = torch.autograd.Variable(grad_old, requires_grad=True)
                                outputs,grad_new = model(inputs_init,inputs_missing,masks,grad_old)
                                
                            elif model.OptimType == 2:
                                hidden          = torch.autograd.Variable(hidden, requires_grad=True)
                                cell            = torch.autograd.Variable(cell, requires_grad=True)
                                outputs,hidden_new,cell_new = model(inputs_init,inputs_missing,masks,hidden,cell)
                                
                            else:                               
                                outputs = model(inputs_init,inputs_missing,masks)
                           
                            # minimized loss types
                            dB          = 0
                            dT          = outputs.size(3)
                            loss_R      = torch.sum( (outputs[:,:,:,dB:dT-dB] - inputs_missing[:,:,:,dB:dT-dB])**2 * masks[:,:,:,dB:dT-dB] )
                            loss_R      = loss_R / torch.sum( masks[:,:,:,dB:dT-dB] )
                            loss_All    = torch.mean( (outputs[:,:,:,dB:dT-dB] - targets_GT[:,:,:,dB:dT-dB])**2 )                            
                            loss_AE     = torch.mean( (model.model_AE(outputs)[:,:,:,dB:dT-dB] - outputs[:,:,:,dB:dT-dB])**2 )                                                        
                            loss_AE_GT  = torch.mean( (model.model_AE(targets_GT)[:,:,:,dB:dT-dB] - targets_GT[:,:,:,dB:dT-dB])**2 )
                            
                            # loss dataType == 1 (data with known groudn-truth)
                            loss        = alpha_Grad * loss_R + alpha_AE * loss_AE

                            loss.backward(retain_graph=True)
                            
                            if model.OptimType >= 0 :
                                optimizer.step()
                                #print(model.model_Grad.gradNet1.weight)
                                #grad    = torch.autograd.grad(loss,x,create_graph=True)[0]
                                #grad.retain_grad()
                                #x     = x - delta * grad
                               
                                # copy the ouput in the input tensor for batch 
                                # after a few iterations
                                if( ( epoch >= iterStateUpdate ) & ( np.mod(epoch,10) == 0 ) ):
                                    print('.... Update init state')
                                    idx_batch = idx.detach().numpy().astype(int)
                                      
                                    UpdateInit = 1
                                    if UpdateInit == 0:                                   
                                        x     = inputs_init - delta * inputs_init.grad.data
                                        x_Curr[idx_batch,:,:] = x.detach().clone().view(-1,outputs.size(2),outputs.size(3)).cpu()
                                        
                                        if model.OptimType == 1:
                                            grad_old          = grad_old - delta * grad_old.grad.data                                    
                                            x_grad1[idx_batch,:,:,:] = grad_old.detach().clone().cpu()
                                        elif model.OptimType == 2:
                                            hidden            = hidden - delta * hidden.grad.data
                                            cell              = cell - delta * cell.grad.data
                                            
                                            x_grad1[idx_batch,:,:] = hidden.detach().clone().cpu()
                                            x_grad2[idx_batch,:,:] = cell.detach().clone().cpu()
                                    elif UpdateInit == 1:                                   
                                        beta = 1.0
                                        x     = (1.-beta) * inputs_init + beta * outputs
                                        x_Curr[idx_batch,:,:] = x.detach().clone().view(-1,outputs.size(2),outputs.size(3)).cpu()
                                        
                                        if model.OptimType == 1:
                                            grad_old          = (1.-beta) * grad_old + beta * grad_new                                   
                                            x_grad1[idx_batch,:,:,:] = grad_old.detach().clone().cpu()
                                        elif model.OptimType == 2:
                                            hidden            = (1.-beta) * hidden + beta * hidden_new                                   
                                            cell              = (1.-beta) * hidden + beta * cell_new                                   
                                            
                                            x_grad1[idx_batch,:,:] = hidden.detach().clone().cpu()
                                            x_grad2[idx_batch,:,:] = cell.detach().clone().cpu()
                            else :
                                idx_batch = idx.detach().numpy().astype(int)
                                x     = inputs_init -  delta * inputs_init.grad.data 
                                x_Curr[idx_batch,:,:] = x.detach().clone().view(-1,outputs.size(2),outputs.size(3)).cpu()
                                                             

                            # other losses
                            loss_I      = torch.sum( (outputs[:,:,:,dB:dT-dB] - targets_GT[:,:,:,dB:dT-dB])**2 * (1. - masks[:,:,:,dB:dT-dB]) )
                            loss_I      = loss_I / torch.sum( (1. - masks[:,:,:,dB:dT-dB]) )
            
                                
                            # statistics
                        
                        running_loss             += loss.item() * inputs_missing.size(0)

                        running_loss_I           += loss_I.item() * inputs_missing.size(0)

                        running_loss_R           += loss_R.item() * inputs_missing.size(0)

                        running_loss_All         += loss_All.item() * inputs_missing.size(0)

                        running_loss_AE          += loss_AE.item() * inputs_missing.size(0)

                        running_loss_AE_GT       += loss_AE_GT.item() * inputs_missing.size(0)

                        num_loss                 += inputs_missing.size(0)
                        #running_expvar += torch.sum( (outputs - inputs)**2 ) / torch.sum(
                    #if phase == 'train':
                    #    exp_lr_scheduler.step()
            
                    epoch_loss        = running_loss / (num_loss)
                    epoch_loss_All    = running_loss_All / num_loss
                    epoch_loss_AE     = running_loss_AE / num_loss
                    epoch_loss_AE_GT  = running_loss_AE_GT / num_loss
                    epoch_loss_I      = running_loss_I / num_loss                                        
                    epoch_loss_R      = running_loss_R / num_loss
                    
                    #epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    epoch_loss     = epoch_loss * stdTr**2
                    epoch_loss_All = epoch_loss_All * stdTr**2
                    epoch_loss_I   = epoch_loss_I * stdTr**2
                    epoch_loss_R   = epoch_loss_R * stdTr**2
                    epoch_loss_AE  = epoch_loss_AE * stdTr**2
                    epoch_loss_AE_GT  = epoch_loss_AE_GT * stdTr**2

                    print('.... Epoch %d/%d: Total loss %.4e [Obs data] NLossAll: %.4e NLossR: %.4e NLossI: %.4e NLossAE: %.4e NLossAEGT: %.4e'%(epoch, num_epochs - 1,epoch_loss,epoch_loss_All,epoch_loss_R,epoch_loss_I,epoch_loss_AE,epoch_loss_AE_GT),flush=True)
           
                    # update dataset in dataloader
                    x_dataset = torch.utils.data.TensorDataset(x_Curr,x_obs,x_mask,x_GT,x_dataType,x_idx,x_grad1,x_grad2) # create your datset
                    dataloaders = {
                            'dataset': torch.utils.data.DataLoader(x_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                            }            
                    # deep copy the model
                    if epoch_loss < best_loss:
                        best_loss      = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())

                # Save model
                if ( flagSaveModel == 1 )  & ( model.OptimType >= 0 ) & ( ( np.mod(epoch,200) == 0  ) | ( epoch == num_epochs - 1) )  :                          
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelAE_iter%03d'%(epoch)+'.mod'
                    print('.................. Auto-Encoder '+fileMod)
                    torch.save(model.model_AE.state_dict(), fileMod)
    
                    fileMod = dirSAVE+genFilename+genSuffixModel+'_modelGrad_iter%03d'%(epoch)+'.mod'
                    print('.................. Gradient model '+fileMod)
                    torch.save(model.model_Grad.state_dict(), fileMod)
                    
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            
        ########################################################################
        ## Apply a trained model for the evaluation of different performance metrics
        ## on training and test datasets
        elif flagProcess[kk] == 10:

            #  use gpu if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(".... Device GPU: "+str(torch.cuda.is_available()))
            print(shapeData.shape)

            # mean-squared error loss
            #criterion = torch.nn.MSELoss()
            var_Tr    = np.var( x_train )
            var_Tt    = np.var( x_test )
                         
            alpha4DVar      = np.array([0.01,1.])
            GradType        = 1 # Gradient computation (0: subgradient, 1: true gradient/autograd)
            OptimType       = 2 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent

            NBGradCurrent   = 20
            NBProjCurrent   = 0
            batch_size      = 8
            
            model           = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
            #model           = NN_4DVar.Model_4DVarNN_Grad(model_AE,shapeData,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
                   
            model           = model.to(device)
            print('4DVar model: Number of trainable parameters = %d'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))
                          
            flagLoadModel   = 1

            # Classic Supervised Learning model using GENN models
            fileAEModelInit = './ResL964DVar/l96_GENN_1_10_02_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_20_modelAE_iter1350.mod'
            #fileAEModelInit = './ResL964DVar/l96_GENN_1_10_02_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_00_20_modelAE_iter550.mod'
            #fileAEModelInit = './ResL964DVar/l96_GENN_1_10_02_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_01_20_modelAE_iter300.mod'
            #fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_01_20_modelAE_iter450.mod'
            
            fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_02_20_modelAE_iter400.mod'
            
            #fileAEModelInit = './ResL964DVar/l96_v3_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_20_modelAE_iter350.mod'
            #fileAEModelInit = './ResL964DVar/l96_v3_GENN_3_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_10_modelAE_iter050.mod'
            
            # GENN-2: using NN_4DVar.Model_4DVarNN_Grad
            #fileAEModelInit = './ResL964DVar/l96_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_20_modelAE_iter400.mod'            
            #fileAEModelInit = './ResL964DVar/l96_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_01_20_modelAE_iter250.mod'
            
            #fileAEModelInit = './ResL964DVar/l96_GENN_2_50_05Flag10__DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_01_20_modelAE_iter000.mod'
            #fileAEModelInit = './ResL964DVar/l96_GENN_2_50_05Flag10__DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_20_modelAE_iter000.mod'
            
            #fileAEModelInit = './ResL964DVar/l96_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_00_02_20_modelAE_iter800.mod'
            #fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_WithObsOnly_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_02_20_modelAE_iter250.mod'
            #fileAEModelInit = './ResL964DVar/l96_v3_GENN_2_50_05_WithObsOnly_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_10_modelAE_iter150.mod'
            
            # models for neurips submission
            #fileAEModelInit = './ResL964DVar/l96_v3_GENN_3_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_20_modelAE_iter315.mod'
            fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_02_20_modelAE_iter250.mod'
            #fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_01_20_modelAE_iter250.mod'
            
            #fileAEModelInit = './ResL964DVar/l96_v3_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_20_modelAE_iter275.mod'
            #fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_WithObsOnly_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_02_20_modelAE_iter250.mod'
            #fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_WithObsOnly_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_02_15_modelAE_iter175.mod'
            fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_WithObsOnly_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_02_20_modelAE_iter250.mod'

            # preprint 4DVar
            fileAEModelInit = './ResL964DVar/l96_v3_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_20_modelAE_iter350.mod'
            fileAEModelInit = './ResL964DVar/l96_v3_GENN_2_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_01_20_modelAE_iter275.mod'
            
            fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_02_20_modelAE_iter400.mod'
            fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_01_20_modelAE_iter250.mod'
            
            fileAEModelInit = './ResL964DVar/l96_v4_DinAE4DVarv1_L96RK4NN_WithObsOnly_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_01_20_modelAE_iter250.mod'
            fileAEModelInit = './ResL964DVar/l96_v4_DinAE4DVarv1_L96RK4NN_WithObsOnly_DT01_200_ObsSubRnd_75_20_NoFTrAE_Nproj00_Grad_01_02_20_modelAE_iter250.mod'
            
            print('.... load model: '+fileAEModelInit)
            model.model_AE.load_state_dict(torch.load(fileAEModelInit))
            model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelAE_iter','_modelGrad_iter')))
             
            #alpha4DVar      = [model.model_Grad.compute_Grad.alphaObs.item(),model.model_Grad.compute_Grad.alphaAE.item()]            
            print('.....')
            print('..... alpha : obs %.3f --- dyn %.3f'%(model.model_Grad.compute_Grad.alphaObs.item(),model.model_Grad.compute_Grad.alphaAE.item()))

            print('.....')
            print('..... alpha4DVar : obs %.3f --- dyn %.3f'%(alpha4DVar[0],alpha4DVar[1]))

            # Create training/test data pytorch tensors and associated  
            # list of tensors (xx[n][x] to access the nth sample for the xth field)
            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init),torch.Tensor(x_train_obs),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_Init),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
            
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }            
            dataset_sizes = {'train': len(training_dataset), 'val': len(test_dataset)}
        

            # training function for dinAE
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:        
                since = time.time()

                model.eval()
                #if phase == 'train':
                #    #rint('Learning')
                #    model.train()  # Set model to training mode
                #else:
                #    #print('Evaluation')
                #    model.eval()   # Set model to evaluate mode
        
                running_loss_Obs     = 0.0
                running_loss_All     = 0.
                running_loss_R       = 0.
                running_loss_I       = 0.
                running_loss_AE      = 0.
                running_loss_AE_GT   = 0.
                running_loss_Obs_GT  = 0.
                num_loss             = 0
        
                # Iterate over data.
                #for inputs_ in dataloaders[phase]:
                #    inputs = inputs_[0].to(device)
                for inputs_init,inputs_missing,masks,targets_GT in dataloaders[phase]:
                    inputs_init    = inputs_init.to(device)
                    inputs_missing = inputs_missing.to(device)
                    masks          = masks.to(device)
                    targets_GT     = targets_GT.to(device)
                    #print(inputs_init.size(0))
        
        
                    # reshaping tensors
                    inputs_init    = inputs_init.view(-1,1,inputs_init.size(1),inputs_init.size(2))
                    inputs_missing = inputs_missing.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                    masks          = masks.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                    targets_GT     = targets_GT.view(-1,1,inputs_init.size(2),inputs_init.size(3))
                                            
                    # forward
                    # need to evaluate grad/backward during the evaluation and training phase for model_AE
                    with torch.set_grad_enabled(True): 
                        #with torch.set_grad_enabled(phase == 'train'):
                        inputs_init    = torch.autograd.Variable(inputs_init, requires_grad=True)
                        if model.OptimType == 1:
                            outputs,grad_new,normgrad = model(inputs_init,inputs_missing,masks,None)
                            
                        elif model.OptimType == 2:
                            outputs,hidden_new,cell_new,normgrad = model(inputs_init,inputs_missing,masks,None,None)
                            
                        else:                               
                            outputs,normgrad = model(inputs_init,inputs_missing,masks)

                        #outputs = model(inputs_init,inputs_missing,masks)
                        
                        loss_R      = torch.sum((outputs - targets_GT)**2 * masks ) / torch.sum(masks)
                        loss_Obs    = torch.sum((outputs - inputs_missing)**2 * masks ) / torch.sum(masks)
                        loss_I      = torch.sum((outputs - targets_GT)**2 * (1. - masks) ) / torch.sum(1.-masks)
                        loss_All    = torch.mean((outputs - targets_GT)**2 )
                        loss_AE     = torch.mean((model.model_AE(outputs) - outputs)**2 )
        
                        loss_AE_GT  = torch.mean((model.model_AE(targets_GT) - targets_GT)**2 )
                        loss_Obs_GT = torch.sum((targets_GT - inputs_missing)**2 * masks ) / torch.sum(masks)
                        #loss_AE_GT  = torch.mean((mod.model_AE(targets_GT) - targets_GT)**2 )
        
                        #loss  = alpha_Grad * loss_All + 0.5 * alpha_AE * ( loss_AE + loss_AE_GT )
                        #if phase == 'train':                                 
                        #    loss  = alpha_Grad * loss_All + 0.5 * alpha_AE * ( loss_AE + loss_AE_GT )
                        #else:
                        #    loss    = alpha_Grad * loss_R + alpha_AE * loss_AE
        
                        # backward + optimize only if in training phase
                                        
                    # statistics
                    running_loss_I           += loss_I.item() * inputs_missing.size(0)
                    running_loss_R           += loss_R.item() * inputs_missing.size(0)
                    running_loss_Obs         += loss_Obs.item() * inputs_missing.size(0)
                    running_loss_Obs_GT      += loss_Obs_GT.item() * inputs_missing.size(0)
                    running_loss_All         += loss_All.item() * inputs_missing.size(0)
                    running_loss_AE_GT       += loss_AE_GT.item() * inputs_missing.size(0)
                    running_loss_AE          += loss_AE.item() * inputs_missing.size(0)
                    num_loss                 += inputs_missing.size(0)
            
                epoch_loss_All   = running_loss_All / num_loss
                epoch_loss_AE    = running_loss_AE / num_loss
                epoch_loss_AE_GT = running_loss_AE_GT / num_loss
                epoch_loss_I     = running_loss_I / num_loss
                epoch_loss_R     = running_loss_R / num_loss
                epoch_loss_Obs   = running_loss_Obs / num_loss
                epoch_loss_Obs_GT= running_loss_Obs_GT / num_loss
                #epoch_acc = running_corrects.double() / dataset_sizes[phase]
    
                epoch_loss_All   = epoch_loss_All * stdTr**2
                epoch_loss_I     = epoch_loss_I * stdTr**2
                epoch_loss_R     = epoch_loss_R * stdTr**2
                epoch_loss_Obs   = epoch_loss_Obs * stdTr**2
                epoch_loss_Obs_GT= epoch_loss_Obs_GT * stdTr**2
                epoch_loss_AE    = epoch_loss_AE * stdTr**2
                epoch_loss_AE_GT = epoch_loss_AE_GT * stdTr**2
                
                epoch_loss_4DVar = alpha4DVar[0] * epoch_loss_Obs + alpha4DVar[1] * epoch_loss_AE
                epoch_loss_4DVar_GT = alpha4DVar[0] * epoch_loss_Obs_GT + alpha4DVar[1] * epoch_loss_AE_GT
    
                print('{} Loss4DVar: {:.4e} Loss4DVarGT: {:.4e} NLossAll: {:.4e} NLossR: {:.4e} NLossI: {:.4e} NLossAE: {:.4e} NLossAEGT: {:.4e}'.format(
                    phase, epoch_loss_4DVar,epoch_loss_4DVar_GT, epoch_loss_All,epoch_loss_R,epoch_loss_I,epoch_loss_AE,epoch_loss_AE_GT),flush=True)
                    #print('... F %f'%model.model_AE.encoder.F)

                print('.... Normalized values ')
                print('{} Loss4DVar: {:.4e} Loss4DVarGT: {:.4e} NLossAll: {:.4e} NLossObsNoisy {:.4e} NLossR: {:.4e} NLossI: {:.4e} NLossAE: {:.4e} NLossAEGT: {:.4e}'.format(
                    phase, epoch_loss_4DVar / stdTr**2,epoch_loss_4DVar_GT / stdTr**2, epoch_loss_All / stdTr**2,epoch_loss_Obs / stdTr**2,epoch_loss_R / stdTr**2,epoch_loss_I / stdTr**2,epoch_loss_AE / stdTr**2,epoch_loss_AE_GT / stdTr**2),flush=True)

                time_elapsed = time.time() - since
                print('Eval. time in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

        ###############################################################
        ## Apply trained model sequentially
        ## Check consistency with L63, normgrad value
        ## Probably not up-to-date
        elif flagProcess[kk] == 11: 
            
            # 4DVAR assimilation
            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # model creation
            alpha           = np.array([1.,0.1])
            GradType        = 1 # Gradient computation (0: subgradient, 1: true gradient/autograd)
            OptimType       = 2 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
            NiterProjection = 0 # Number of fixed-point iterations
            NiterGrad       = 20 # Number of gradient descent step
            
            NBGradCurrent   = NiterGrad
            NBProjCurrent   = NiterProjection
            model           = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
            model           = model.to(device)
            print('4DVar model: Number of trainable parameters = %d'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))

            # Load trained model
            flagLoadModel   = 1
            fileAEModelInit = './ResL634DVar/l63_DinAE4DVar_L63EulerNN_DT01_200_075_Noise10_Nproj01_Grad_01_02_20_modelAE_iter999.mod'
            fileAEModelInit = './ResL964DVar/l96_v3_GENN_3_50_05_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_10_modelAE_iter050.mod'


            if flagLoadModel == 1:
                model.model_AE.load_state_dict(torch.load(fileAEModelInit))
                model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelAE_iter','_modelGrad_iter')))

            # optimizer
            #lrCurrent   = 1e-4  
            #optimizer   = optim.Adam(model.parameters(), lr= lrCurrent)
            #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
            
            # assimilation loop
            t0    = 0
            dt    = 0.01
            NIter = 100
            alpha = 0.99
            delta = 0.2
            
            # Apply current model to data
            x_train_Curr         = np.copy(x_train_Init)
            x_test_Curr          = np.copy(x_test_Init)
            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Curr),torch.Tensor(x_train_obs),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_Curr),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }
            
            x_test_pred  = []

            phase = 'val'
            compt = 0
            numData = 0

            loss_dyn1 = torch.Tensor([0.0])
            loss_obs1 = torch.Tensor([0.0])
            loss_GT1  = torch.Tensor([0.0])
            
            for x_init,x_obs,masks,x_GT in dataloaders[phase]:
                x_init    = x_init.to(device)
                x_obs     = x_obs.to(device)
                masks     = masks.to(device)
                x_GT      = x_GT.to(device)
                               
                if compt < 1:

                    model.eval()
                         
                    compt    = compt + 1
                    numData +=  x_GT.size(0)
                    print('Process batch %d/%d'%(compt,int(x_test.shape[0]/batch_size)), flush=True)
                    
                    ## apply global model
                    ## just for checking
                    xhat1 = model(x_init,x_obs,masks)

                    loss_dyn1 = loss_dyn1 + x_GT.size(0) * torch.mean((xhat1 - model.model_AE( xhat1 ))**2 )
                    loss_obs1 = loss_obs1 + x_GT.size(0) * torch.sum((xhat1 - x_obs)**2 * masks) / torch.sum( masks)
                    loss_GT1  = loss_GT1 + x_GT.size(0) * torch.mean((xhat1 - x_GT)**2 )
                    
                    # AE projection
                    xhat = x_init 
                    for kk in range(0,NiterProjection):
                        xhat = model.model_AE( xhat )
                        xhat = xhat * (1. - masks) + x_obs * masks
                
                    # gradient normalisation
                    grad     = model.model_Grad.compute_Grad(xhat, model.model_AE(xhat),x_obs,masks)
                    normgrad = torch.sqrt( torch.mean( grad**2 ) )

                    # 4DVar assimilation using trained gradient descent
                    losses_test = []
                    compt_kk = 0
                    for kk in range(0,NIter):
                        # AE projection
                        x_pred = model.model_AE( xhat )
                        
                        # Compute Gradient-based update
                        if model.OptimType == 0:
                            grad  = model.model_Grad( xhat, x_pred, x_obs, masks, normgrad)
                        elif model.OptimType == 1:
                            if kk == 0:
                                grad  = model.model_Grad( xhat, x_pred, x_obs, masks, None, normgrad)
                            else:
                                grad  = model.model_Grad( xhat, x_pred, x_obs, masks, grad_old, normgrad)
                            grad_old = 1. * grad
                        elif model.OptimType == 2:
                            if kk == 0:
                                grad,hidden,cell  = model.model_Grad( xhat, x_pred, x_obs, masks, None,None, normgrad)
                            else:
                                grad,hidden,cell  = model.model_Grad( xhat, x_pred, x_obs, masks, hidden, cell, normgrad)
                            
                        xhat = xhat - grad
                     
                        # evaluate and store losses
                        if( np.mod(kk,1) == 0 ):  
                             # losses
                            loss_dyn = torch.mean((xhat - model.model_AE( xhat ))**2)
                            #loss_dyn = torch.sum((xhat - model.model_AE( xhat ))**2  , dim = -1)
                            #loss_dyn = torch.sum( loss_dyn  , dim = -1)
        
                            loss_obs = torch.sum((xhat - x_obs)**2 * masks) / torch.sum( masks)
                            #loss_obs = torch.sum((xhat - x_obs)**2 * masks , dim = -1)
                            #loss_obs = torch.sum( loss_obs  , dim = -1)
        
                            loss_GT = torch.mean((xhat - x_GT)**2)

                            # store as np
                            #if( compt_kk == 0 ):
                            #    losses_test = torch.cat( (loss_GT.view(1,-1,1),loss_dyn.view(1,-1,1),loss_obs.view(1,-1,1)), dim = 2 ).cpu().detach().numpy()                                
                            #else:
                            #    losses_test = np.concatenate( (losses_test,torch.cat( (loss_GT.view(1,-1,1),loss_dyn.view(1,-1,1),loss_obs.view(1,-1,1)), dim = 2 ).cpu().detach().numpy()) , axis = 1)
                            #compt_kk = compt_kk + 1
                            #print('..... Loss %d:  GT %.6e   ---  dyn %.6e   ---  obs %.6e'%(kk,losses_test[0,-1,0],losses_test[0,-1,1],losses_test[0,-1,2]) )
                        
                        # evaluate and store losses
                        #if( np.mod(kk,500) == 0 ):  
    
                            # store as np
                            if( compt_kk == 0 ):
                                losses_test = torch.cat( (loss_GT.view(1,1),loss_dyn.view(1,1),loss_obs.view(1,1)), dim = 1 ).cpu().detach().numpy()                                
                            else:
                                losses_test = np.concatenate( (losses_test,torch.cat( (loss_GT.view(1,1),loss_dyn.view(1,1),loss_obs.view(1,1)), dim = 1 ).cpu().detach().numpy()) , axis = 0)
                            compt_kk = compt_kk + 1
                    if compt == 1 :
                        loss4DVar = x_GT.size(0) * losses_test
                    else :
                        loss4DVar = loss4DVar + x_GT.size(0) * losses_test


            loss_dyn1 = stdTr * loss_dyn1 / numData
            loss_obs1 = stdTr * loss_obs1 / numData
            loss_GT1  = stdTr * loss_GT1  / numData
            
            lossNN    = np.array([loss_GT1.cpu().detach().numpy(),loss_dyn1.cpu().detach().numpy(),loss_obs1.cpu().detach().numpy()])            
            loss4DVar = stdTr * loss4DVar / numData
                            
            print('..........................................')
            print('.....')
            print('..... Loss (trained model) #d:  GT %.6e   ---  dyn %.6e   ---  obs %.6e'%(lossNN[0],lossNN[1],lossNN[2]))
            print('.....')
            print('')
            for kk in range(0,loss4DVar.shape[0]):
                print('..... Loss trained Grad Descent %d:  GT %.6e   ---  dyn %.6e   ---  obs %.6e'%(kk,loss4DVar[kk,0],loss4DVar[kk,0],loss4DVar[kk,0]) )

         ###############################################################
        ## 4DVAR Assimilation given a trained AE model 
        ## using a fixed-step gradient descent
        elif flagProcess[kk] == 12: 
            print('......................................................................')
            print('......................................................................')
            print('..........    4DVar assimilation using a fixed-step gradient descent  ')
            print('..........')
            print('..........')
            # 4DVAR assimilation
            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # model creation
            GradType        = 1 # Gradient computation (0: subgradient, 1: true gradient/autograd)
            OptimType       = 2 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
            NiterProjection = 0 # Number of fixed-point iterations
            NiterGrad       = 20 # Number of gradient descent step
            
            NBGradCurrent   = NiterGrad
            NBProjCurrent   = NiterProjection
            #model           = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
            model           = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
            model           = model.to(device)

            # Load trained model
            flagLoadModel   = 0
            #fileAEModelInit = './ResL634DVar/l63_DinAE4DVar_L63EulerNN_DT01_200_075_Noise10_Nproj01_Grad_01_02_20_modelAE_iter999.mod'
            #fileAEModelInit = './ResL634DVar/l63_DinAE4DVar_L63EulerNN_DT01_200_ObsSub_87_20_Nproj01_Grad_01_02_20_modelAE_iter400.mod'
            
            fileAEModelInit = './ResL964DVar/l96_v3_DinAE4DVarv1_L96RK4NN_WithObsOnly_DT01_200_ObsSubRnd_75_20_Nproj00_Grad_01_02_20_modelAE_iter350.mod'
            if flagLoadModel == 1:
                model.model_AE.load_state_dict(torch.load(fileAEModelInit))
                model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelAE_iter','_modelGrad_iter')))

            # optimizer
            #lrCurrent   = 1e-4  
            #optimizer   = optim.Adam(model.parameters(), lr= lrCurrent)
            #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
            
            # assimilation loop
            #t0    = 0
            #dt    = 0.01
            NIter      = 50001
            alpha4DVar = 1e0* np.array([.01,1.])
            delta      = 2e4
            #batch_size = 256
            
            # Apply current model to data
            x_train_Curr         = np.copy(x_train_Init)
            x_test_Curr          = np.copy(x_test_Init)
            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Curr),torch.Tensor(x_train_obs),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_Curr),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }
            
            x_test_pred  = []

            phase   = 'val'
            compt   = 0
            numData = 0
            
            loss_dyn1 = torch.Tensor([0.0])
            loss_obs1 = torch.Tensor([0.0])
            loss_GT1  = torch.Tensor([0.0])            
            
            for x_init,x_obs,masks,x_GT in dataloaders[phase]:
                x_init    = x_init.to(device)
                x_obs     = x_obs.to(device)
                masks     = masks.to(device)
                x_GT      = x_GT.to(device)
                
                        # reshaping tensors
                x_init   = x_init.view(-1,1,x_init.size(1),x_init.size(2))
                x_obs    = x_obs.view(-1,1,x_obs.size(1),x_obs.size(2))
                masks    = masks.view(-1,1,masks.size(1),masks.size(2))
                x_GT     = x_GT.view(-1,1,x_GT.size(1),x_GT.size(2))

                if 1*1: #compt < 2: ##  ## 
                    #for phase in ['train', 'val']:    
                    model.eval()
                         
                    compt    = compt + 1
                    numData +=  x_GT.size(0)
                    print('Process batch %d/%d'%(compt,int(x_test.shape[0]/batch_size)), flush=True)

                    ## apply trained model AE + Grad for comparison
                    with torch.set_grad_enabled(True): 
                        #with torch.set_grad_enabled(phase == 'train'):
                        x_init    = torch.autograd.Variable(x_init, requires_grad=True)
                        if model.OptimType == 1:
                            xhat1,grad_new,normgrad = model(x_init,x_obs,masks,None)
                            
                        elif model.OptimType == 2:
                            xhat1,hidden_new,cell_new,normgrad = model(x_init,x_obs,masks,None,None)
                            
                        else:                               
                            xhat1,normgrad = model(x_init,x_obs,masks)
                        
                    loss_dyn1 = loss_dyn1 + x_GT.size(0) * torch.mean((xhat1 - model.model_AE( xhat1 ))**2 )
                    loss_obs1 = loss_obs1 + x_GT.size(0) * torch.sum((xhat1 - x_obs)**2 * masks) / torch.sum( masks)
                    loss_GT1  = loss_GT1 + x_GT.size(0) * torch.mean((xhat1 - x_GT)**2 )
                    
                    # AE projection
                    xhat = x_init 
                    xhat = torch.autograd.Variable(xhat, requires_grad=True)
    
                    # 4DVar assimilation using trained gradient descent
                    losses_test = []
                    compt_kk = 0
                    for kk in range(0,NIter):
                        # AE projection
                        x_pred = model.model_AE( xhat )
                        
                        # dynamical loss
                        loss_dyn = torch.mean((xhat - x_pred)**2 )
                    
                        # observation loss
                        loss_obs = torch.sum((xhat - x_obs)**2 * masks) / torch.sum( masks)
                    
                        ## loss wrt groudn-truth
                        loss_GT  = torch.mean((xhat - x_GT)**2)
    
                        # overall loss
                        loss = alpha4DVar[0] * loss_obs + alpha4DVar[1] * loss_dyn 
                    
                        # compute gradient w.r.t. X and update X
                        loss.backward()
                        
                        #grad_X  = torch.autograd.grad(loss,X_torch,create_graph=True)
                        xhat = xhat - delta * xhat.grad.data
                        xhat = torch.autograd.Variable(xhat, requires_grad=True)
                                                                         
                        # evaluate and store losses
                        if( np.mod(kk,500) == 0 ):  
    
                            # store as np
                            if( compt_kk == 0 ):
                                losses_test = torch.cat( (loss.view(1,1),loss_GT.view(1,1),loss_dyn.view(1,1),loss_obs.view(1,1)), dim = 1 ).cpu().detach().numpy()                                
                            else:
                                losses_test = np.concatenate( (losses_test,torch.cat( (loss.view(1,1),loss_GT.view(1,1),loss_dyn.view(1,1),loss_obs.view(1,1)), dim = 1 ).cpu().detach().numpy()) , axis = 0)
                            compt_kk = compt_kk + 1
                    if compt == 1 :
                        loss4DVar = x_GT.size(0) * losses_test
                    else :
                        loss4DVar = loss4DVar + x_GT.size(0) * losses_test

            loss_dyn1 = stdTr**2 * loss_dyn1 / numData
            loss_obs1 = stdTr**2 * loss_obs1 / numData
            loss_GT1  = stdTr**2 * loss_GT1  / numData
            
            lossNN    = np.array([loss_GT1.cpu().detach().numpy(),loss_dyn1.cpu().detach().numpy(),loss_obs1.cpu().detach().numpy()])            
            loss4DVar = stdTr**2 * loss4DVar / numData
                            
            print('..........................................')
            print('.....')
            print('..... Loss (trained Grad model) #d:  GT %.6e   ---  dyn %.6e   ---  obs %.6e'%(lossNN[0],lossNN[1],lossNN[2]))
            print('.....')
            print('..... alpha : obs %.3f --- dyn %.3f'%(alpha4DVar[0],alpha4DVar[1]))

            for kk in range(0,loss4DVar.shape[0]):
                print('..... Loss Grad Descent %d:  4DVar %.6e   ---  GT %.6e   ---  dyn %.6e   ---  obs %.6e'%(500*kk,loss4DVar[kk,0],loss4DVar[kk,1],loss4DVar[kk,2],loss4DVar[kk,3]) )

        ###############################################################
        ## 4DVAR Assimilation given a trained AE model 
        ## using a fixed-step gradient descent
        elif flagProcess[kk] == 13: 
            
            # 4DVAR assimilation
            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # model creation
            alpha4DVar      = np.array([0.001,1.])#np.array([0.30,1.60])#
            GradType        = 1 # Gradient computation (0: subgradient, 1: true gradient/autograd)
            OptimType       = 1 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
            NiterProjection = 1 # Number of fixed-point iterations
            NiterGrad       = 5 # Number of gradient descent step
            
            NBGradCurrent   = NiterGrad
            NBProjCurrent   = NiterProjection
            model           = NN_4DVar.Model_4DVarNN_GradFP(model_AE,shapeData,NBProjCurrent,NBGradCurrent,GradType,OptimType,InterpFlag,UsePriodicBoundary)        
            model           = model.to(device)

            # Load trained model
            flagLoadModel   = 0
            #fileAEModelInit = './ResL634DVar/l63_DinAE4DVar_L63EulerNN_DT01_200_075_Noise10_Nproj01_Grad_01_02_20_modelAE_iter999.mod'
            #fileAEModelInit = './ResL634DVar/l63_DinAE4DVar_L63EulerNN_DT01_200_ObsSub_87_20_Nproj01_Grad_01_02_20_modelAE_iter400.mod'
            
            fileAEModelInit = './ResL964DVar/l96_GENN_4_10_05_DT01_200_ObsSub_75_20_NoFTrAE_Nproj01_Grad_01_01_05_modelAE_iter000.mod'
            if flagLoadModel == 1:
                model.model_AE.load_state_dict(torch.load(fileAEModelInit))
                #model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelAE_iter','_modelGrad_iter')))

            # optimizer
            #lrCurrent   = 1e-4  
            #optimizer   = optim.Adam(model.parameters(), lr= lrCurrent)
            #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
            
            # assimilation loop
            #t0    = 0
            #dt    = 0.01
            NIter = 300001#25001
            delta = 1e3 # alphaObs = 0.001: 1e3, NIter 250001 alphaObs = 0.01: 1e3, NIter 250001 --- alphaObs = 0.1: 5e2, NIter 250001
            batch_size = 64#
            delta = delta * batch_size
            
            
            alpha4DVar[0] = alpha4DVar[0] / alpha4DVar[1]
            alpha4DVar[1] = 1.0
            
            print('... alphaObs %.3f'%alpha4DVar[0])    
            print('... alphaPrior %.3f'%alpha4DVar[1])    
            print('... delta %.3f'%(delta/batch_size))

            #NIter = 20001
            #alpha = 0.95
            #delta = 50. * x_train_Init.shape[2]
            #batch_size = 256
            
            # Apply current model to data
            x_train_Curr         = np.copy(x_train_Init)
            x_test_Curr          = np.copy(x_test_Init)
            training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Curr),torch.Tensor(x_train_obs),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset
            test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_Curr),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) # create your datset
            dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }
            
            x_test_pred  = []

            phase   = 'val'
            compt   = 0
            numData = 0
            
            loss_dyn1 = torch.Tensor([0.0])
            loss_obs1 = torch.Tensor([0.0])
            loss_GT1  = torch.Tensor([0.0])
            
            
            loss_dyn_GT = torch.Tensor([0.0])
            loss_obs_GT = torch.Tensor([0.0])

            for x_init,x_obs,masks,x_GT in dataloaders[phase]:
                x_init    = x_init.to(device)
                x_obs     = x_obs.to(device)
                masks     = masks.to(device)
                x_GT      = x_GT.to(device)
                
                        # reshaping tensors
                x_init   = x_init.view(-1,1,x_init.size(1),x_init.size(2))
                x_obs    = x_obs.view(-1,1,x_obs.size(1),x_obs.size(2))
                masks    = masks.view(-1,1,masks.size(1),masks.size(2))
                x_GT     = x_GT.view(-1,1,x_GT.size(1),x_GT.size(2))

                if 1*1: # compt < 1:#  
                    #for phase in ['train', 'val']:    
                    model.eval()
                         
                    compt    = compt + 1
                    numData +=  x_GT.size(0)
                    print('Process batch %d/%d'%(compt,int(x_test.shape[0]/batch_size)), flush=True)

                    ## apply trained model AE + Grad for comparison
                    xhat1 = model(x_init,x_obs,masks)
                    xhat1 = xhat1[0]
                        
                    loss_dyn1 = loss_dyn1 + x_GT.size(0) * torch.mean((xhat1 - model.model_AE( xhat1 ))**2 )
                    loss_obs1 = loss_obs1 + x_GT.size(0) * torch.sum((xhat1 - x_obs)**2 * masks) / torch.sum( masks)
                    loss_GT1  = loss_GT1 + x_GT.size(0) * torch.mean((xhat1 - x_GT)**2 )
                    
                    ## apply to true state
                    loss_dyn_GT = loss_dyn_GT + x_GT.size(0) * torch.mean((x_GT - model.model_AE( x_GT ))**2 )
                    loss_obs_GT = loss_obs_GT + x_GT.size(0) * torch.sum((x_GT - x_obs)**2 * masks) / torch.sum( masks)

                    # AE projection
                    xhat = x_init 
                    xhat = torch.autograd.Variable(xhat, requires_grad=True)
    
                    # 4DVar assimilation using trained gradient descent
                    losses_test = []
                    compt_kk = 0
                    for kk in range(0,NIter):
                        # AE projection
                        x_pred = model.model_AE( xhat )
                        
                        # dynamical loss
                        loss_dyn = torch.mean((xhat - x_pred)**2 )
                    
                        # observation loss
                        loss_obs = torch.sum((xhat - x_obs)**2 * masks) / torch.sum( masks)
                    
                        ## loss wrt groudn-truth
                        loss_GT  = torch.mean((xhat - x_GT)**2)
    
                        # overall loss
                        loss = alpha4DVar[0] * loss_obs + alpha4DVar[1] * loss_dyn 
                    
                        # compute gradient w.r.t. X and update X
                        loss.backward()
                        
                        #grad_X  = torch.autograd.grad(loss,X_torch,create_graph=True)
                        xhat = xhat - delta * xhat.grad.data
                        xhat = torch.autograd.Variable(xhat, requires_grad=True)
                                                                         
                        # evaluate and store losses
                        if( np.mod(kk,1000) == 0 ):  
    
                            # store as np
                            if( compt_kk == 0 ):
                                losses_test = torch.cat( (loss_GT.view(1,1),loss_dyn.view(1,1),loss_obs.view(1,1)), dim = 1 ).cpu().detach().numpy()                                
                            else:
                                losses_test = np.concatenate( (losses_test,torch.cat( (loss_GT.view(1,1),loss_dyn.view(1,1),loss_obs.view(1,1)), dim = 1 ).cpu().detach().numpy()) , axis = 0)
                            compt_kk = compt_kk + 1
                    if compt == 1 :
                        loss4DVar = x_GT.size(0) * losses_test
                    else :
                        loss4DVar = loss4DVar + x_GT.size(0) * losses_test

            loss_dyn1 = stdTr**2 * loss_dyn1 / numData
            loss_obs1 = stdTr**2 * loss_obs1 / numData
            loss_GT1  = stdTr**2 * loss_GT1  / numData
            
            loss_dyn_GT = stdTr**2 * loss_dyn_GT / numData
            loss_obs_GT = stdTr**2 * loss_obs_GT / numData

            lossNN    = np.array([loss_GT1.cpu().detach().numpy(),loss_dyn1.cpu().detach().numpy(),loss_obs1.cpu().detach().numpy()])            
            loss4DVar = stdTr**2 * loss4DVar / numData
                            
            print('..........................................')
            print('.....')
            print('..... Loss (trained Grad model) #d:  GT %.6e   ---  dyn %.6e   ---  obs %.6e'%(lossNN[0],lossNN[1],lossNN[2]))
            print('..... 4DVar loss true state:  %.6e'%(alpha4DVar[0] * loss_obs_GT  + alpha4DVar[1] * loss_dyn_GT ))
            print('.....')
            print('')
            for kk in range(0,loss4DVar.shape[0]):
                print('..... Loss Grad Descent %d:  GT %.6e   ---  dyn %.6e   ---  obs %.6e   ---  all %.6e'%(1000*kk,loss4DVar[kk,0],loss4DVar[kk,1],loss4DVar[kk,2],alpha4DVar[1]*loss4DVar[kk,1]+alpha4DVar[0]*loss4DVar[kk,2]) )