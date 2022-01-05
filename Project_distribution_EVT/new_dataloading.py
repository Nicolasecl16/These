import numpy as np
import pytorch_lightning as pl
import xarray as xr
#from torch.utils.data import Dataset, ConcatDataset, DataLoader
from netCDF4 import Dataset

import scipy
import torch
import datetime
from sklearn.feature_extraction import image
from models import LitModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_size          = 1000
N=data_size
batch_size  = 64

DT = 10
DX = 30

Obs_noise = 0.001
Model_noise = 0.02 #pour f_sum 0.02

Database = torch.zeros(N,DX,DT)


#######################################################################################################################
#Définition des fonctions de simulation dynamique :

#Modèle de somme et erreur additive
def f_sum(x) :
    I = torch.rand(N,DX)
    x[:,:,0] = I
    for i in range(N) :
        for k in range(DX):
            for j in range(1,DT):
                if j==1:
                    x[i,k,j] = 0.25*x[i,k,j-1]+0.25*x[i,(k-1)%DX,j-1]+0.25*x[i,(k+1)%DX,j-1]+0.25*x[i,(k+2)%DX,j-1]+0.25*x[i,(k-2)%DX,j-1] +Model_noise*np.random.randn()
                else : 
                    x[i,k,j] = 0.25*x[i,k,j-1]+0.25*x[i,(k-1)%DX,j-1]+0.25*x[i,(k+1)%DX,j-1]+0.25*x[i,(k+2)%DX,j-2]+0.25*x[i,(k-2)%DX,j-2] +Model_noise*np.random.randn()
    return(x)

# modèle de somme avec erreur multiplicative
def f_sum2(x) :
    I = torch.rand(N,DX)
    x[:,:,0] = I
    for i in range(N) :
        for k in range(DX):
            for j in range(1,DT):
                if j==1:
                    x[i,k,j] = 0.6*x[i,k,j-1]+0.4*x[i,(k-1)%DX,j-1]+0.2*x[i,(k+1)%DX,j-1]+0.1*x[i,(k+2)%DX,j-1]+0.05*x[i,(k-2)%DX,j-1]+ (x[i,k,j-1]+x[i,(k-1)%DX,j-1])*Model_noise*np.random.randn()
                else : 
                    x[i,k,j] = 0.6*x[i,k,j-1]+0.4*x[i,(k-1)%DX,j-1]+0.2*x[i,(k+1)%DX,j-1]+0.1*x[i,(k+2)%DX,j-2]+0.05*x[i,(k-2)%DX,j-2] +(x[i,k,j-1]+x[i,(k-1)%DX,j-1])*Model_noise*np.random.randn()
    return(x)


                
Database = f_sum2(Database)  

dataTrainingNoNaND = Database[:int(0.6*N),:,:]
dataValNoNaND = Database[int(0.6*N):int(0.8*N),:,:]
dataTestNoNaND = Database[int(0.8*N):,:,:]


# flagTypeMissData = 0 : reconstruction
# flagTypeMissData = 1 : prévision

flagTypeMissData = 1


if flagTypeMissData ==0 :    
    rateMissingData = 0.5#0.9
           
# create missing data
#flagTypeMissData = 0 : Missing data randomly chosen on the patch driven by rateMissingData
#flagTpeMissData = 1  : Prevision

if flagTypeMissData == 0:
    indRandD         = np.random.permutation(dataTrainingNoNaND.shape[0]*dataTrainingNoNaND.shape[1]*dataTrainingNoNaND.shape[2])
    indRandD         = indRandD[0:int(rateMissingData*len(indRandD))]
    dataTrainingD    = torch.tensor(dataTrainingNoNaND).reshape((dataTrainingNoNaND.shape[0]*dataTrainingNoNaND.shape[1]*dataTrainingNoNaND.shape[2],1))
    dataTrainingD[indRandD] = float('nan')
    dataTrainingD    = torch.reshape(dataTrainingD,(dataTrainingNoNaND.shape[0],dataTrainingNoNaND.shape[1],dataTrainingNoNaND.shape[2]))
            
    indRandD         = np.random.permutation(dataValNoNaND.shape[0]*dataValNoNaND.shape[1]*dataValNoNaND.shape[2])
    indRandD         = indRandD[0:int(rateMissingData*len(indRandD))]
    dataValD    = torch.tensor(dataValNoNaND).reshape((dataValNoNaND.shape[0]*dataValNoNaND.shape[1]*dataValNoNaND.shape[2],1))
    dataValD[indRandD] = float('nan')
    dataValD    = np.reshape(dataValD,(dataValNoNaND.shape[0],dataValNoNaND.shape[1],dataValNoNaND.shape[2]))
            
            
    indRandD         = np.random.permutation(dataTestNoNaND.shape[0]*dataTestNoNaND.shape[1]*dataTestNoNaND.shape[2])
    indRandD         = indRandD[0:int(rateMissingData*len(indRandD))]
    dataTestD        = torch.tensor(dataTestNoNaND).reshape((dataTestNoNaND.shape[0]*dataTestNoNaND.shape[1]*dataTestNoNaND.shape[2],1))
    dataTestD[indRandD] = float('nan')
    dataTestD          = np.reshape(dataTestD,(dataTestNoNaND.shape[0],dataTestNoNaND.shape[1],dataTestNoNaND.shape[2]))

    genSuffixObs    = '_ObsRnd_%02d_%02d'%(100*rateMissingData,10*Model_noise**2)
 
else :
    frac_prev = 2/3  #Proportion de jours dont on a les observations
    
    dataTrainingD    = torch.zeros((dataTrainingNoNaND.shape))
    dataTrainingD[:] = float('nan')
            
    dataValD    = torch.zeros((dataValNoNaND.shape))
    dataValD[:] = float('nan')
            
    dataTestD        = torch.zeros((dataTestNoNaND.shape))
    dataTestD[:]     = float('nan')
    
    dataTrainingD[:,:,:(int(2*DT/3+1))] = dataTrainingNoNaND[:,:,:(int(frac_prev*DT+1))]
    dataValD[:,:,:(int(2*DT/3+1))] = dataValNoNaND[:,:,:(int(frac_prev*DT+1))]
    dataTestD[:,:,:(int(2*DT/3+1))] = dataTestNoNaND[:,:,:(int(frac_prev*DT+1))]
    genSuffixObs    = '_ObsSubRnd_%02d_%02d'%(100*Obs_noise,10*Model_noise**2)
    
print('... Data type: '+genSuffixObs)
    #for nn in range(0,dataTraining.shape[1],time_step_obs):
    #    dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
    #dataTest    = np.zeros((dataTestNoNaN.shape))
    #dataTest[:] = float('nan')
    #dataTest[:,::time_step_obs,:] = dataTestNoNaN[:,::time_step_obs,:]

            
                                
# mask for NaN
maskTrainingD = (dataTrainingD == dataTrainingD).type_as(dataTrainingD)
maskValD = (dataValD == dataValD).type_as(dataTrainingD)
maskTestD     = ( dataTestD    ==  dataTestD   ).type_as(dataTrainingD)
            
dataTrainingD = torch.nan_to_num(dataTrainingD)       
dataValD = torch.nan_to_num(dataValD)
dataTestD     = torch.nan_to_num(dataTestD)
            
    # Permutation to have channel as #1 component
dataTrainingD      = torch.moveaxis(dataTrainingD,-1,1)
maskTrainingD      = torch.moveaxis(maskTrainingD,-1,1)
dataTrainingNoNaND = torch.moveaxis(dataTrainingNoNaND,-1,1)
        
dataValD      = torch.moveaxis(dataValD,-1,1)
maskValD      = torch.moveaxis(maskValD,-1,1)
dataValNoNaND = torch.moveaxis(dataValNoNaND,-1,1)
            
dataTestD      = torch.moveaxis(dataTestD,-1,1)
maskTestD      = torch.moveaxis(maskTestD,-1,1)
dataTestNoNaND = torch.moveaxis(dataTestNoNaND,-1,1)
            
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

if flagTypeMissData == 0:    
    mean_Tr= torch.mean(X_train_missingD[:],0)/torch.mean(mask_trainD,0)
    std_Tr =torch.std((X_train_missingD-mean_Tr)*mask_trainD,0)/torch.sqrt((torch.mean(mask_trainD,0)))

    x_train_missingD = (X_train_missingD - mean_Tr*mask_trainD)*(1/std_Tr*mask_trainD)           
    x_val_missingD = (X_val_missingD - mean_Tr*mask_valD)*(1/std_Tr*mask_valD)
    x_test_missingD  = (X_test_missingD - mean_Tr*mask_testD)*(1/std_Tr*mask_testD)

    x_trainD = (X_trainD - mean_Tr) / std_Tr
    x_valD = (X_valD - mean_Tr) / std_Tr
    x_testD  = (X_testD - mean_Tr) / std_Tr

    # Generate noisy observsation
        
    X_train_obsD = X_train_missingD + Obs_noise * maskTrainingD * torch.randn(X_train_missingD.shape[0],X_train_missingD.shape[1],X_train_missingD.shape[2])
    X_val_obsD = X_val_missingD + Obs_noise * maskValD * torch.randn(X_val_missingD.shape[0],X_val_missingD.shape[1],X_val_missingD.shape[2])
    X_test_obsD  = X_test_missingD  + Obs_noise * maskTestD * torch.randn(X_test_missingD.shape[0],X_test_missingD.shape[1],X_test_missingD.shape[2])

    mean_Tr_obs= torch.mean(X_train_obsD[:],0)/torch.mean(mask_trainD,0)
    std_Tr_obs = torch.std((X_train_obsD-mean_Tr_obs)*mask_trainD,0)/torch.sqrt((torch.mean(mask_trainD,0)))
            
    x_train_obsD = (X_train_obsD - mean_Tr_obs*mask_trainD)*(1/std_Tr_obs*mask_trainD)
    x_val_obsD = (X_val_obsD - mean_Tr_obs*mask_valD)*(1/std_Tr_obs*mask_valD)
    x_test_obsD  = (X_test_obsD - mean_Tr_obs*mask_testD)*(1/std_Tr_obs*mask_testD)

    
elif flagTypeMissData==1 : 
    mean_Tr= torch.mean(X_trainD[:],0)
    std_Tr =torch.std((X_trainD-mean_Tr),0)

    x_train_missingD = (X_train_missingD - mean_Tr*mask_trainD)/std_Tr
    x_val_missingD = (X_val_missingD - mean_Tr*mask_valD)/std_Tr
    x_test_missingD  = (X_test_missingD - mean_Tr*mask_testD)/std_Tr

    x_trainD = (X_trainD - mean_Tr) /std_Tr
    x_valD = (X_valD - mean_Tr) / std_Tr
    x_testD  = (X_testD - mean_Tr) / std_Tr

    # Generate noisy observsation
        
    X_train_obsD = X_train_missingD + Obs_noise * maskTrainingD * torch.randn(X_train_missingD.shape[0],X_train_missingD.shape[1],X_train_missingD.shape[2])
    X_val_obsD = X_val_missingD + Obs_noise * maskValD * torch.randn(X_val_missingD.shape[0],X_val_missingD.shape[1],X_val_missingD.shape[2])
    X_test_obsD  = X_test_missingD  + Obs_noise * maskTestD * torch.randn(X_test_missingD.shape[0],X_test_missingD.shape[1],X_test_missingD.shape[2])

    mean_Tr_obs= torch.mean(X_train_obsD[:],0)
    std_Tr_obs = torch.nan_to_num(torch.std((X_train_obsD-mean_Tr_obs),0)/torch.sqrt((torch.mean(mask_trainD,0))),1)
            
    x_train_obsD = (X_train_obsD - mean_Tr_obs)/std_Tr_obs
    x_val_obsD = (X_val_obsD - mean_Tr_obs)/std_Tr_obs
    x_test_obsD  = (X_test_obsD - mean_Tr_obs*mask_testD)/std_Tr_obs

          
print('..... Training dataset: %dx%dx%d'%(X_train_missingD.shape[0],X_trainD.shape[1],X_trainD.shape[2]))
print('..... Validation dataset: %dx%dx%d'%(X_valD.shape[0],X_valD.shape[1],X_valD.shape[2]))
print('..... Test dataset    : %dx%dx%d'%(X_testD.shape[0],X_testD.shape[1],X_testD.shape[2]))
            

print('........ Initialize interpolated states')
## Initial interpolation
#flagInit = 0 : Input values for the forced normal distribution
#flagInit = 1 : Input values for the forced GPD distribution
#flagInit = 2 : Input values with pre-calculated covariance

flagInit = 2
   
if flagInit == 0: 
    
    
    if flagTypeMissData == 0:
        X_train_InitD = torch.zeros(int(0.6*N),DT,DX,2)
        X_train_InitD[:,:,:,1] =torch.ones(int(0.6*N),DT,DX)
        X_val_InitD = torch.zeros(int(0.2*N),DT,DX,2)
        X_val_InitD[:,:,:,1] =torch.ones(int(0.2*N),DT,DX)
        X_test_InitD = torch.zeros(int(0.2*N),DT,DX,2)
        X_test_InitD[:,:,:,1] =torch.ones(int(0.2*N),DT,DX)
    
        for ii in range(0,X_trainD.shape[0]):
        # Initial linear interpolation for each component
            XInitD = torch.zeros((X_trainD.shape[1],X_trainD.shape[2]))
           
            for kk in range(0,mask_trainD.shape[1]):
                indt  = torch.where( mask_trainD[ii,kk,:] == 1.0 )[0]
                indt_ = torch.where( mask_trainD[ii,kk,:] == 0.0 )[0]
           
                if len(indt) > 1:
                    indt_[ torch.where( indt_ <torch.min(indt)) ] = torch.min(indt)
                    indt_[ torch.where( indt_ > torch.max(indt)) ] = torch.max(indt)
                    fkk = scipy.interpolate.interp1d(indt, X_train_obsD[ii,kk,indt])
                    XInitD[kk,indt]  = X_train_obsD[ii,kk,indt]
                    XInitD[kk,indt_] = torch.Tensor(fkk(indt_))
                else:
                    XInitD = XInitD + meanTr
            
            X_train_InitD[ii,:,:,0] = XInitD
            
        for ii in range(0,X_valD.shape[0]):
            # Initial linear interpolation for each component
            XInitD = torch.zeros((X_valD.shape[1],X_valD.shape[2]))
           
            for kk in range(0,mask_valD.shape[1]):
                indt  = torch.where( mask_valD[ii,kk,:] == 1.0 )[0]
                indt_ = torch.where( mask_valD[ii,kk,:] == 0.0 )[0]
           
                if len(indt) > 1:
                    indt_[ torch.where( indt_ < torch.min(indt)) ] = torch.min(indt)
                    indt_[ torch.where( indt_ > torch.max(indt)) ] = torch.max(indt)
                    fkk = scipy.interpolate.interp1d(indt, X_val_obsD[ii,kk,indt])
                    XInitD[kk,indt]  = X_val_obsD[ii,kk,indt]
                    XInitD[kk,indt_] = torch.Tensor(fkk(indt_))
                else:
                    XInitD = XInitD + meanTr
            
            X_val_InitD[ii,:,:,0] = XInitD
            

        for ii in range(0,X_testD.shape[0]):
        # Initial linear interpolation for each component
            XInit = torch.zeros((X_testD.shape[1],X_testD.shape[2]))
            
            for kk in range(0,X_testD.shape[1]):
                indt  = torch.where( mask_testD[ii,kk,:] == 1.0 )[0]
                indt_ = torch.where( mask_testD[ii,kk,:] == 0.0 )[0]
            
                if len(indt) > 1:
                    indt_[ torch.where( indt_ < torch.min(indt)) ] = torch.min(indt)
                    indt_[ torch.where( indt_ > torch.max(indt)) ] = torch.max(indt)
                    fkk = scipy.interpolate.interp1d(indt, X_test_obsD[ii,kk,indt])
                    XInit[kk,indt]  = X_test_obsD[ii,kk,indt]
                    XInit[kk,indt_] = torch.Tensor(fkk(indt_))
                else:
                    XInit = XInit + meanTr
        
            X_test_InitD[ii,:,:,0] = XInit

        
    elif flagTypeMissData == 1:
        Cov_known_Init = 1
        Cov_unknown_Init = 1
        
        X_train_InitD = torch.zeros(int(0.6*N),DT,DX,2)
        X_train_InitD[:,:,:,1] = torch.ones(int(0.6*N),DT,DX)
        X_train_InitD[:,:int(frac_prev*DT+1),:,1] =Cov_known_Init
        X_train_InitD[:,int(frac_prev*DT+1):,:,1] =Cov_unknown_Init
        X_val_InitD = torch.zeros(int(0.2*N),DT,DX,2)
        X_val_InitD[:,:,:,1] =torch.ones(int(0.2*N),DT,DX)
        X_val_InitD[:,:int(frac_prev*DT+1),:,1] =Cov_known_Init
        X_val_InitD[:,int(frac_prev*DT+1):,:,1] =Cov_unknown_Init
        X_test_InitD = torch.zeros(int(0.2*N),DT,DX,2)
        X_test_InitD[:,:,:,1] =torch.ones(int(0.2*N),DT,DX)
        X_test_InitD[:,:int(frac_prev*DT+1),:,1] =Cov_known_Init
        X_test_InitD[:,int(frac_prev*DT+1):,:,1] =Cov_unknown_Init
        X_train_InitD[:,:,:,0] = X_train_obsD
        XInitD = X_train_InitD[:,int(frac_prev*DT),:,0]
        for i in range(int(frac_prev*DT)+1,DT):
            X_train_InitD[:,i,:,0] = XInitD
            
        X_val_InitD[:,:,:,0] = X_val_obsD
        XInitD = X_val_InitD[:,int(frac_prev*DT),:,0]
        for i in range(int(frac_prev*DT)+1,DT):
            X_val_InitD[:,i,:,0] = XInitD
        
        X_test_InitD[:,:,:,0] = X_test_obsD
        XInitD = X_test_InitD[:,int(frac_prev*DT),:,0]
        for i in range(int(frac_prev*DT)+1,DT):
            X_test_InitD[:,i,:,0] = XInitD
            
        
        
elif flagInit == 2 :
    if flagTypeMissData == 1:
        ckpt_path ='Res_Update/modelsum-Exp2-epoch=243-val_loss=-7.84.ckpt'
        lit_cls = LitModel
        mod = lit_cls.load_from_checkpoint(ckpt_path, 
                                                    mean_Tr=0, 
                                                    std_Tr = 0
                                                    )
        mod=mod.to(device)
    
        Cov_known_Init = 1
        Cov_unknown_Init = 1
        
        X_train_InitD = torch.zeros(int(0.6*N),DT,DX,2)
        X_train_InitD[:,:,:,1] = torch.ones(int(0.6*N),DT,DX)
        X_train_InitD[:,:int(frac_prev*DT+1),:,1] =Cov_known_Init
        X_train_InitD[:,int(frac_prev*DT+1):,:,1] =Cov_unknown_Init
        X_val_InitD = torch.zeros(int(0.2*N),DT,DX,2)
        X_val_InitD[:,:,:,1] =torch.ones(int(0.2*N),DT,DX)
        X_val_InitD[:,:int(frac_prev*DT+1),:,1] =Cov_known_Init
        X_val_InitD[:,int(frac_prev*DT+1):,:,1] =Cov_unknown_Init
        X_test_InitD = torch.zeros(int(0.2*N),DT,DX,2)
        X_test_InitD[:,:,:,1] =torch.ones(int(0.2*N),DT,DX)
        X_test_InitD[:,:int(frac_prev*DT+1),:,1] =Cov_known_Init
        X_test_InitD[:,int(frac_prev*DT+1):,:,1] =Cov_unknown_Init
        X_train_InitD[:,:,:,0] = X_train_obsD
        XInitD = X_train_InitD[:,int(frac_prev*DT),:,0]
        for i in range(int(frac_prev*DT)+1,DT):
            X_train_InitD[:,i,:,0] = XInitD
            
        X_val_InitD[:,:,:,0] = X_val_obsD
        XInitD = X_val_InitD[:,int(frac_prev*DT),:,0]
        for i in range(int(frac_prev*DT)+1,DT):
            X_val_InitD[:,i,:,0] = XInitD
        
        X_test_InitD[:,:,:,0] = X_test_obsD
        XInitD = X_test_InitD[:,int(frac_prev*DT),:,0]
        for i in range(int(frac_prev*DT)+1,DT):
            X_test_InitD[:,i,:,0] = XInitD
        X_train_InitD.requires_grad =True
        X_train_InitD = X_train_InitD
        X_val_InitD.requires_grad =True
        X_test_InitD.requires_grad =True
        X_train_InitD = mod.model(X_train_InitD.to(device),X_train_obsD.to(device),mask_trainD.to(device))[0]
        X_val_InitD = mod.model(X_val_InitD.to(device), X_val_obsD.to(device),mask_valD.to(device))[0]
        X_test_InitD = mod.model(X_test_InitD.to(device),X_test_obsD.to(device),mask_testD.to(device))[0]
        X_train_InitD[:,:,:,1] = torch.abs(X_train_InitD[:,:,:,1])
        X_val_InitD[:,:,:,1] = torch.abs(X_val_InitD[:,:,:,1])
        X_test_InitD[:,:,:,1] = torch.abs(X_test_InitD[:,:,:,1]) 
        X_train_InitD = X_train_InitD.detach().cpu()
        X_val_InitD = X_val_InitD.detach().cpu()
        X_test_InitD = X_test_InitD.detach().cpu()
           
#x_train_InitD = torch.zeros(X_train_InitD.shape[0],DT,DX,2)
#x_val_InitD = torch.zeros(X_val_InitD.shape[0],DT,DX,2)
#x_test_InitD = torch.zeros(X_test_InitD.shape[0],DT,DX,2) 

#x_train_InitD[:,:,:,0] = ( X_train_InitD[:,:,:,0] - mean_Tr ) / std_Tr
#x_val_InitD[:,:,:,0] = ( X_val_InitD[:,:,:,0] - mean_Tr ) / std_Tr
#x_test_InitD[:,:,:,0] = ( X_test_InitD[:,:,:,0] - mean_Tr ) / std_Tr
#x_train_InitD[:,:,:,1] = X_train_InitD[:,:,:,1]                
#x_val_InitD[:,:,:,1] = X_val_InitD[:,:,:,1] 
#x_test_InitD[:,:,:,1] = X_test_InitD[:,:,:,1] 

print('..... Training dataset: %dx%dx%d'%(X_trainD.shape[0],X_trainD.shape[1],X_trainD.shape[2]))
print('..... Validation dataset: %dx%dx%d'%(X_valD.shape[0],X_valD.shape[1],X_valD.shape[2]))
print('..... Test dataset    : %dx%dx%d'%(X_testD.shape[0],X_testD.shape[1],X_testD.shape[2]))
    
    
training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(X_train_InitD),torch.Tensor(X_train_obsD),torch.Tensor(mask_trainD),torch.Tensor(X_trainD)) # create your datset
print(type(X_train_InitD))
print(X_train_InitD.dtype)

val_dataset         = torch.utils.data.TensorDataset(torch.Tensor(X_val_InitD),torch.Tensor(X_val_obsD),torch.Tensor(mask_valD),torch.Tensor(X_valD)) # create your datset
     
test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(X_test_InitD),torch.Tensor(X_test_obsD),torch.Tensor(mask_testD),torch.Tensor(X_testD)) # create your datset

        
dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }            
dataset_sizes = {'train': len(training_dataset),'val': len(val_dataset), 'test': len(test_dataset)}



