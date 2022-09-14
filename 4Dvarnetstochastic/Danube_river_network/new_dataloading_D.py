import numpy as np
import pytorch_lightning as pl
import xarray as xr
#from torch.utils.data import Dataset, ConcatDataset, DataLoader
from netCDF4 import Dataset

import torch
import datetime
from sklearn.feature_extraction import image
import scipy

NbDays          = 18244
  
batch_size  = 64#4#4#8#12#8#256#8 originellement 96

flagRandomSeed = 0
    ###############################################################
    ## data generation including noise sampling and missing data gaps       
print('........ Data generation')
if flagRandomSeed == 0:
    print('........ Random seed set to 100')
    np.random.seed(100)


time_step  = 1
DT = 48
sigNoise   = np.sqrt(2)
rateMissingData = 0.75#0.9
width_med_filt_spatial = 5
width_med_filt_temp = 1


# loss weghing wrt time
w_ = np.zeros(DT)
w_[int(DT / 2)] = 1.
wLoss = torch.Tensor(w_)

flagTypeMissData = 3
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
while (i+1)*550<(NbDays-1):
    x=550*i
    Indtrain.append([x,(x+350)])
    Indval.append([(x+350),(x+450)])
    Indtest.append([x+450,x+550])
    i+=1
    


day0=datetime.date(1960,1,1)
dayend=datetime.date(2009,12,12)



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
#flagTypeMissData = 3 : The stations listed in MaskedStations are masked, obs available every 4 days
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
    Rate_space =0.5
    R_permute=np.random.permutation(31)
    MaskedStations = R_permute[:int(Rate_space*31)]
    print('Masked Stations')
    print(MaskedStations)
    AvailableStations = R_permute[int(Rate_space*31):]        
    dataTrainingD    = np.zeros((dataTrainingNoNaND.shape))
    dataTrainingD[:] = float('nan')
            
    dataValD    = np.zeros((dataValNoNaND.shape))
    dataValD[:] = float('nan')
            
    dataTestD        = np.zeros((dataTestNoNaND.shape))
    dataTestD[:]     = float('nan')
    Availabletimestep = np.arange(0,DT+1,int(1/(1-rateMissingData)))
    Availabletimestep[-1] = DT-1
    for i in AvailableStations :
        dataTrainingD[:,Availabletimestep,i]= dataTrainingNoNaND[:,Availabletimestep,i]
        dataValD[:,Availabletimestep,i]= dataValNoNaND[:,Availabletimestep,i]
        dataTestD[:,Availabletimestep,i] = dataTestNoNaND[:,Availabletimestep,i]
        genSuffixObs    = '_ObsSubRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
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
    print('data shape')
    print(X_train_missingD.shape)
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
        stdTr[i] =1
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


print(meanTr.shape)
print(stdTr.shape)

print('..... Training dataset: %dx%dx%d'%(x_train_missingD.shape[0],x_trainD.shape[1],x_trainD.shape[2]))
print('..... Validation dataset: %dx%dx%d'%(x_valD.shape[0],x_valD.shape[1],x_valD.shape[2]))
print('..... Test dataset    : %dx%dx%d'%(x_testD.shape[0],x_testD.shape[1],x_testD.shape[2]))
            

print('........ Initialize interpolated states')
## Initial interpolation
#flagInit = 0 : Masked values are replaced by 0
#flagInit = 1 : Masked values are replaced by last available value (prevision)
#flaginit = 2 : Interpolation 
#flagInit = 3 : Input contains values for the parameters of the distribution, mu initialisé à 0, ksi à 0.05 et sigma à 1

flagInit = 2

            
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
    
    
    
    
elif flagInit == 2:
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
        
elif flagInit == 3:
    X_ext_train = X_train_missingD[:,:,int(2*DT/3)].reshape(X_train_missingD.shape[0],X_train_missingD.shape[1],1)
    X_train_InitD = mask_trainD * X_train_obsD + (1. - mask_trainD)*X_ext_train
    X_ext_val = X_val_missingD[:,:,int(2*DT/3)].reshape(X_val_missingD.shape[0],X_val_missingD.shape[1],1)
    X_val_InitD = mask_valD * X_val_obsD + (1. - mask_valD)*X_ext_val
    X_ext_test = X_test_missingD[:,:,int(2*DT/3)].reshape(X_test_missingD.shape[0],X_test_missingD.shape[1],1)
    X_test_InitD = mask_testD * X_test_obsD + (1. - mask_testD)*X_ext_test
    C = np.zeros(X_train_Init.shape[0],X_train_Init.shape[1],3*(DT-1)+1)
    C[:,:,0] = 1
    C[:,:,1:13] = 0
    C[:,:,13:25] = 0.1
    C[:,:,25:37] = 1
    Xtrain_InitD = 0
                        
x_train_InitD = ( X_train_InitD - meanTr ) / stdTr
x_val_InitD = ( X_val_InitD - meanTr ) / stdTr
x_test_InitD = ( X_test_InitD - meanTr ) / stdTr
        
# reshape to dT for time dimension
DT = DT
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

if flagInit==2: 
    DX = X_train_InitD.shape[1]

    Xt                 = np.zeros((X_train_InitD.shape[0],2*DX,X_train_InitD.shape[2]))
    xt                 = np.zeros((X_train_InitD.shape[0],2*DX,X_train_InitD.shape[2]))
    Xt[:,DX:,:]        += 1
    xt[:,DX:,:]        += 1
    Xt[:,:DX,:]         = X_train_InitD
    xt[:,:DX,:]         = x_train_InitD
    x_train_InitD       = xt
    X_train_InitD       = Xt

    Xv                 = np.zeros((X_val_InitD.shape[0],2*DX,X_val_InitD.shape[2]))
    xv                 = np.zeros((X_val_InitD.shape[0],2*DX,X_val_InitD.shape[2]))
    Xv[:,DX:,:]        += 1
    xv[:,DX:,:]        += 1    
    Xv[:,:DX,:]         = X_val_InitD
    xv[:,:DX,:]         = x_val_InitD
    x_val_InitD         = xv
    X_val_InitD         = Xv
           
    Xtt                 = np.zeros((X_test_InitD.shape[0],2*DX,X_test_InitD.shape[2]))
    xtt                 = np.zeros((X_test_InitD.shape[0],2*DX,X_test_InitD.shape[2]))
    Xtt[:,DX:,:]        += 1
    xtt[:,DX:,:]        += 1     
    Xtt[:,:DX,:]         = X_test_InitD
    xtt[:,:DX,:]         = x_test_InitD
    x_test_InitD         = xtt
    X_test_InitD         = Xtt

print('..... Training dataset: %dx%dx%d'%(x_trainD.shape[0],x_trainD.shape[1],x_trainD.shape[2]))
print('..... Validation dataset: %dx%dx%d'%(x_valD.shape[0],x_valD.shape[1],x_valD.shape[2]))
print('..... Test dataset    : %dx%dx%d'%(x_testD.shape[0],x_testD.shape[1],x_testD.shape[2]))

print(x_train_InitD[0,:,:])
print(x_train_obsD)
    
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

if flagTypeMissData == 4 : 
    X_val    =  X_valD[:,:,:]
    #val_mseRec = compute_metrics(X_val,mod.x_rec_debit)            
    #val_norm_rmse_debit = np.sqrt( np.mean( X_val.ravel()**2 ) )
            
else : 
    X_val    =  dataValNoNaND[:,:,int(DT/2)]
        
        ## postprocessing
        #if width_med_filt_spatial + width_med_filt_temp > 2 :
            #mod.x_rec_debit = ndimage.median_filter(mod.x_rec_debit,size=(width_med_filt_temp,width_med_filt_spatial,width_med_filt_spatial))
        
    #val_mseRec = compute_metrics(X_val,mod.x_rec_debit)     
       
    #val_norm_rmse_debit = np.sqrt( np.mean( X_val.ravel()**2 ) )

    #print('\n\n........................................ ')
    #print('........................................\n ')
    #trainer.test(mod, test_dataloaders=dataloaders['test'])
        
        
if flagTypeMissData==4 :
    X_Test   =  X_testD[:,:,:]
            #X_mask = []
        
    #test_mseRec = compute_metrics(X_Test,mod.x_rec_debit)     
        
        
    test_norm_rmse_debit = np.sqrt( np.mean( X_Test.ravel()**2 ) )

            #saveRes = True# 
    debit_gt = X_Test
        
    debit_obs = X_test_missingD[:,:,:]
        
    #debit_rec = mod.x_rec_debit    
            
else:     
    X_Test   =  dataTestNoNaND[:,:,int(DT/2)]
    #X_mask = []
        
    #test_mseRec = compute_metrics(X_Test,mod.x_rec_debit)     
        
        
    test_norm_rmse_debit = np.sqrt( np.mean( X_Test.ravel()**2 ) )

    saveRes = True# 
    debit_gt = X_Test
        
    debit_obs = X_test_missingD[:,:,int(DT/2)]
        
            #debit_rec = mod.x_rec_debit            
            
    

class XrDataset(Dataset):
    """
    torch Dataset based on an xarray file with on the fly slicing.
    """

    def __init__(self, path, var, slice_win, dim_range=None, strides=None, decode=False, resize_factor=1):
        """
        :param path: xarray file
        :param var: data variable to fetch
        :param slice_win: window size for each dimension {<dim>: <win_size>...}
        :param dim_range: Optional dimensions bounds for each dimension {<dim>: slice(<min>, <max>)...}
        :param strides: strides on each dim while scanning the dataset {<dim>: <dim_stride>...}
        :param decode: Whether to decode the time dim xarray (useful for gt dataset)
        """
        super().__init__()

        self.var = var
        _ds = xr.open_dataset(path)
        if decode:
            _ds.time.attrs["units"] = "seconds since 2012-10-01"
            _ds = xr.decode_cf(_ds)
        self.ds = _ds.sel(**(dim_range or {}))
        if resize_factor!=1:
            self.ds = self.ds.coarsen(lon=resize_factor).mean(skipna=True).coarsen(lat=resize_factor).mean(skipna=True)
        self.slice_win = slice_win
        self.strides = strides or {}
        self.ds_size = {
            dim: max((self.ds.dims[dim] - slice_win[dim]) // self.strides.get(dim, 1) + 1, 0)
            for dim in slice_win
        }

    def __del__(self):
        self.ds.close()

    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __getitem__(self, item):
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx,
                       self.strides.get(dim, 1) * idx + self.slice_win[dim])
            for dim, idx in zip(self.ds_size.keys(),
                                np.unravel_index(item, tuple(self.ds_size.values())))
        }
        return self.ds.isel(**sl)[self.var].data.astype(np.float32)


class FourDVarNetDataset(Dataset):
    """
    Dataset for the 4DVARNET method:
        an item contains a slice of OI, mask, and GT
        does the preprocessing for the item
    """

    def __init__(
            self,
            slice_win,
            dim_range=None,
            strides=None,
            oi_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc',
            oi_var='ssh_mod',
            obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data_new/dataset_nadir_0d_swot.nc',
            obs_mask_var='ssh_mod',
            # obs_mask_var='mask',
            gt_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
            gt_var='ssh',
            sst_path=None,
            sst_var=None,
            resize_factor=1,
    ):
        super().__init__()

        self.oi_ds = XrDataset(oi_path, oi_var, slice_win=slice_win, dim_range=dim_range, strides=strides, resize_factor=resize_factor)
        self.gt_ds = XrDataset(gt_path, gt_var, slice_win=slice_win, dim_range=dim_range, strides=strides, decode=True, resize_factor=resize_factor)
        self.obs_mask_ds = XrDataset(obs_mask_path, obs_mask_var, slice_win=slice_win, dim_range=dim_range,
                                     strides=strides, resize_factor=resize_factor)

        self.norm_stats = None

        if sst_var is not None:
            self.sst_ds = XrDataset(sst_path, sst_var, slice_win=slice_win,
                                    dim_range=dim_range, strides=strides,
                                    decode=sst_var == 'sst', resize_factor=resize_factor)
        else:
            self.sst_ds = None
        self.norm_stats_sst = None

    def set_norm_stats(self, stats, stats_sst=None):
        self.norm_stats = stats
        self.norm_stats_sst = stats_sst

    def __len__(self):
        return min(len(self.oi_ds), len(self.gt_ds), len(self.obs_mask_ds))

    def __getitem__(self, item):
        mean, std = self.norm_stats
        _oi_item = self.oi_ds[item]
        _oi_item = (np.where(
            np.abs(_oi_item) < 10,
            _oi_item,
            np.nan,
        ) - mean) / std

        _gt_item = (self.gt_ds[item] - mean) / std
        oi_item = np.where(~np.isnan(_oi_item), _oi_item, 0.)
        # obs_mask_item = self.obs_mask_ds[item].astype(bool) & ~np.isnan(oi_item) & ~np.isnan(_gt_item)
        _obs_item = self.obs_mask_ds[item]
        obs_mask_item = ~np.isnan(_obs_item)
        obs_item = np.where(~np.isnan(_obs_item), _obs_item, np.zeros_like(_obs_item))

        gt_item = _gt_item

        if self.sst_ds == None:
            return oi_item, obs_mask_item, obs_item, gt_item
        else:
            mean, std = self.norm_stats_sst
            _sst_item = (self.sst_ds[item] - mean) / std
            sst_item = np.where(~np.isnan(_sst_item), _sst_item, 0.)

            return oi_item, obs_mask_item, obs_item, gt_item, sst_item

class FourDVarNetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            slice_win,
            dim_range=None,
            strides=None,
            train_slices=(slice('2012-10-01', "2012-11-20"), slice('2013-02-07', "2013-09-30")),
            # train_slices=(slice('2012-10-01', "2012-10-10"),),
            test_slices=(slice('2013-01-03', "2013-01-27"),),
            val_slices=(slice('2012-11-30', "2012-12-24"),),
            oi_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/oi/ssh_NATL60_swot_4nadir.nc',
            oi_var='ssh_mod',
            obs_mask_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/data/dataset_nadir_0d_swot.nc',
            obs_mask_var='ssh_mod',
            # obs_mask_var='mask',
            gt_path='/gpfsstore/rech/yrf/commun/NATL60/NATL/ref/NATL60-CJM165_NATL_ssh_y2013.1y.nc',
            gt_var='ssh',
            sst_path=None,
            sst_var=None,
            resize_factor=1,
            dl_kwargs=None,
    ):
        super().__init__()

        self.resize_factor = resize_factor
        self.dim_range = dim_range
        self.slice_win = slice_win
        self.strides = strides
        self.dl_kwargs = {
            **{'batch_size': 4, 'num_workers': 2, 'pin_memory': True},
            **(dl_kwargs or {})
        }

        self.oi_path = oi_path
        self.oi_var = oi_var
        self.obs_mask_path = obs_mask_path
        self.obs_mask_var = obs_mask_var
        self.gt_path = gt_path
        self.gt_var = gt_var
        self.sst_path = sst_path
        self.sst_var = sst_var

        self.resize_factor = resize_factor

        self.train_slices, self.test_slices, self.val_slices = train_slices, test_slices, val_slices
        self.train_ds, self.val_ds, self.test_ds = None, None, None
        self.norm_stats = None
        self.norm_stats_sst = None

    def compute_norm_stats(self, ds):
        mean = float(xr.concat([_ds.gt_ds.ds[_ds.gt_ds.var] for _ds in ds.datasets], dim='time').mean())
        std = float(xr.concat([_ds.gt_ds.ds[_ds.gt_ds.var] for _ds in ds.datasets], dim='time').std())

        if self.sst_var == None:
            return mean, std
        else:
            print('... Use SST data')
            mean_sst = float(xr.concat([_ds.sst_ds.ds[_ds.sst_ds.var] for _ds in ds.datasets], dim='time').mean())
            std_sst = float(xr.concat([_ds.sst_ds.ds[_ds.sst_ds.var] for _ds in ds.datasets], dim='time').std())

            return [mean, std], [mean_sst, std_sst]

    def set_norm_stats(self, ds, ns, ns_sst=None):
        for _ds in ds.datasets:
            _ds.set_norm_stats(ns, ns_sst)

    def get_domain_bounds(self, ds):
        min_lon = round(np.min(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])), 2)
        max_lon = round(np.max(np.concatenate([_ds.gt_ds.ds['lon'].values for _ds in ds.datasets])), 2)
        min_lat = round(np.min(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])), 2)
        max_lat = round(np.max(np.concatenate([_ds.gt_ds.ds['lat'].values for _ds in ds.datasets])), 2)
        return min_lon, max_lon, min_lat, max_lat

    def get_domain_split(self):
        return self.test_ds.datasets[0].gt_ds.ds_size

    def setup(self, stage=None):
        self.train_ds, self.val_ds, self.test_ds = [
            ConcatDataset(
                [FourDVarNetDataset(
                    dim_range={**self.dim_range, **{'time': sl}},
                    strides=self.strides,
                    slice_win=self.slice_win,
                    oi_path=self.oi_path,
                    oi_var=self.oi_var,
                    obs_mask_path=self.obs_mask_path,
                    obs_mask_var=self.obs_mask_var,
                    gt_path=self.gt_path,
                    gt_var=self.gt_var,
                    sst_path=self.sst_path,
                    sst_var=self.sst_var,
                    resize_factor=self.resize_factor,
                ) for sl in slices]
            )
            for slices in (self.train_slices, self.val_slices, self.test_slices)
        ]

        if self.sst_var == None:
            self.norm_stats = self.compute_norm_stats(self.train_ds)
            self.set_norm_stats(self.train_ds, self.norm_stats)
            self.set_norm_stats(self.val_ds, self.norm_stats)
            self.set_norm_stats(self.test_ds, self.norm_stats)
        else:
            self.norm_stats, self.norm_stats_sst = self.compute_norm_stats(self.train_ds)

            self.set_norm_stats(self.train_ds, self.norm_stats, self.norm_stats_sst)
            self.set_norm_stats(self.val_ds, self.norm_stats, self.norm_stats_sst)
            self.set_norm_stats(self.test_ds, self.norm_stats, self.norm_stats_sst)

        self.bounding_box = self.get_domain_bounds(self.train_ds)
        self.ds_size = self.get_domain_split()

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self.dl_kwargs, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, **self.dl_kwargs, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, **self.dl_kwargs, shuffle=False)


if __name__ == '__main__':
    """
    Test run for single batch loading and trainer.fit 
    """

    # Specify the dataset spatial bounds
    dim_range = {
        'lat': slice(35, 45),
        'lon': slice(-65, -55),
    }

    # Specify the batch patch size
    slice_win = {
        'time': 5,
        'lat': 200,
        'lon': 200,
    }
    # Specify the stride between two patches
    strides = {
        'time': 1,
        'lat': 200,
        'lon': 200,
    }

    dm = FourDVarNetDataModule(
        slice_win=slice_win,
        dim_range=dim_range,
        strides=strides,
    )

    # Test a single batch loading
    dm.setup()
    dl = dm.val_dataloader()
    batch = next(iter(dl))
    oi, mask, gt = batch

    # Test fit
    from main import LitModel

    lit_mod = LitModel()
    trainer = pl.Trainer(gpus=1)
    # dm.setup()
    trainer.fit(lit_mod, datamodule=dm)
