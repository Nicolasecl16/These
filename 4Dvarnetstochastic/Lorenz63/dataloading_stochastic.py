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

import scipy
from scipy.integrate import solve_ivp
import sdeint
import torch

batch_size = 32  

flagRandomSeed = 0
    ###############################################################
    ## data generation including noise sampling and missing data gaps       
print('........ Data generation')
if flagRandomSeed == 0:
    print('........ Random seed set to 100')
    np.random.seed(100)

class GD:
    model = 'Lorenz_63'
    class parameters:
        sigma = 10.0
        rho = 28.0
        beta = 8.0/3
        gamma = 2
    dt_integration = 0.01 # integration time
    dt_states = 1 # number of integeration times between consecutive states (for xt and catalog)
    dt_obs = 8 # number of integration times between consecutive observations (for yo)
    var_obs = np.array([0,1,2]) # indices of the observed variables
    nb_loop_train = 10**2 # size of the catalog
    nb_loop_test = 20000 # size of the true state and noisy observations
    sigma2_catalog = 1.0 # variance of the model error to generate the catalog
    sigma2_obs = 2.0 # variance of the observation error to generate observation

class time_series:
  values = 0.
  time   = 0.

    
def Lorenz63s(z,t,sigma,rho,beta,gamma):
    """ Lorenz-63 dynamical model. 
    Args:
        z: the state. shape = (3,)
        t: time
        sigma, rho, beta, gamma: the parameters of L63s
    Returns:
        dzdt: dz/dt
    """
    x_1 = sigma*(z[1]-z[0])-4*z[0]/(2*gamma)
    x_2 = z[0]*(rho-z[2])-z[1]-4*z[0]/(2*gamma)
    x_3 = z[0]*z[1] - beta*z[2] -8*z[2]/(2*gamma)
    dzdt  = np.array([x_1,x_2,x_3])
    
    return dzdt

def Stoch_Lorenz_63(S,t,
                    sigma=GD.parameters.sigma, 
                    rho = GD.parameters.rho, 
                    beta = GD.parameters.beta, 
                    gamma = GD.parameters.gamma):
    """ Lorenz-63 dynamical model. """
    x_1 = sigma*(S[1]-S[0])-4*S[0]/(2*gamma);
    x_2 = S[0]*(rho-S[2])-S[1] -4*S[1]/(2*gamma);
    x_3 = S[0]*S[1] - beta*S[2] -8*S[2]/(2*gamma);
    dS  = np.array([x_1,x_2,x_3]);
    return dS

def brownian_process(S,t,
                     sigma=GD.parameters.sigma, 
                     rho = GD.parameters.rho,
                     beta = GD.parameters.beta,
                     gamma = GD.parameters.gamma):
    x_1 = 0.0;
    x_2 = (rho - S[2])/np.sqrt(gamma);
    x_3 = (S[1])/np.sqrt(gamma);
    dS  = np.array([x_1,x_2,x_3]);
    G = np.eye((3))
    np.fill_diagonal(G,dS)
    return G    
    
def AnDA_Lorenz_63(S,t,sigma,rho,beta):
#Lorenz-63 dynamical model. 
    x_1 = sigma*(S[1]-S[0]);
    x_2 = S[0]*(rho-S[2])-S[1];
    x_3 = S[0]*S[1] - beta*S[2];
    dS  = np.array([x_1,x_2,x_3]);
    return dS




## data generation: L63 series
GD = GD()    
y0 = np.array([8.0,0.0,30.0])
S = sdeint.itoEuler(Stoch_Lorenz_63, 
                          brownian_process, 
                          y0 = y0, 
                          tspan = np.arange(0,5+0.000001,GD.dt_integration)
                        )
print(S[-1])                         
y0 = S[-1];
S = sdeint.itoEuler(Stoch_Lorenz_63, 
                          brownian_process, 
                          y0 = y0, 
                          tspan = np.arange(0,GD.nb_loop_test*GD.dt_integration+0.000001,GD.dt_integration)
                        )


              
####################################################
## Generation of training and test dataset
## Extraction of time series of dT time steps  
NbTraining = 10000
NbTest     = 2000
NbVal      = 2000
time_step = 1
dT        = 200
sigNoise  = np.sqrt(2.0)
rateMissingData = (1-1./8.)#0.75#0.95
  
xt = time_series()
xt.values = S
xt.time   = np.arange(0,GD.nb_loop_test*GD.dt_integration+0.000001,GD.dt_integration)
# extract subsequences
dataTrainingNoNaN = image.extract_patches_2d(xt.values[0:12000:time_step,:],(dT,3),NbTraining)
#dataValNoNaN = image.extract_patches_2d(xt.values[?:time_step,:],(dT,GD.parameters.J),NbVal)
print(dataTrainingNoNaN.shape)
dataValNoNaN     = image.extract_patches_2d(xt.values[15000::time_step,:],(dT,3),NbVal)
print(dataValNoNaN.shape)
dataTestNoNaN     = image.extract_patches_2d(xt.values[13000::time_step,:],(dT,3),NbTest)
print(dataTestNoNaN.shape)
print(np.mean(dataTrainingNoNaN))

# create missing data
flagTypeMissData = 1

if flagTypeMissData == 0:
    indRand         = np.random.permutation(dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2])
    indRand         = indRand[0:int(rateMissingData*len(indRand))]
    dataTraining    = np.copy(dataTrainingNoNaN).reshape((dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2],1))
    dataTraining[indRand] = float('nan')
    dataTraining    = np.reshape(dataTraining,(dataTrainingNoNaN.shape[0],dataTrainingNoNaN.shape[1],dataTrainingNoNaN.shape[2]))
    
    indRand         = np.random.permutation(dataValNoNaN.shape[0]*dataValNoNaN.shape[1]*dataValNoNaN.shape[2])
    indRand         = indRand[0:int(rateMissingData*len(indRand))]
    dataVal        = np.copy(dataValNoNaN).reshape((dataValNoNaN.shape[0]*dataValNoNaN.shape[1]*dataValNoNaN.shape[2],1))
    dataVal[indRand] = float('nan')
    dataVal        = np.reshape(dataVal,(dataValNoNaN.shape[0],dataValNoNaN.shape[1],dataValNoNaN.shape[2]))        
    
    indRand         = np.random.permutation(dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2])
    indRand         = indRand[0:int(rateMissingData*len(indRand))]
    dataTest        = np.copy(dataTestNoNaN).reshape((dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2],1))
    dataTest[indRand] = float('nan')
    dataTest          = np.reshape(dataTest,(dataTestNoNaN.shape[0],dataTestNoNaN.shape[1],dataTestNoNaN.shape[2]))
    
    genSuffixObs    = '_ObsRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
else:
    time_step_obs   = int(1./(1.-rateMissingData))
    dataTraining    = np.zeros((dataTrainingNoNaN.shape))
    dataTraining[:] = float('nan')
    dataTraining[:,::time_step_obs,0] = dataTrainingNoNaN[:,::time_step_obs,0]
    
    dataVal        = np.zeros((dataValNoNaN.shape))
    dataVal[:]     = float('nan')
    dataVal[:,::time_step_obs,0] = dataValNoNaN[:,::time_step_obs,0]
    
    dataTest        = np.zeros((dataTestNoNaN.shape))
    dataTest[:]     = float('nan')
    dataTest[:,::time_step_obs,0] = dataTestNoNaN[:,::time_step_obs,0]
                
    if 1*0:
        dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
        dataVal[:,::time_step_obs,:]     = dataValNoNaN[:,::time_step_obs,:]
        dataTest[:,::time_step_obs,:]     = dataTestNoNaN[:,::time_step_obs,:]
                    
        genSuffixObs    = '_ObsSub_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
    elif 1*0:
        for nn in range(0,dataTraining.shape[1],time_step_obs):
            indrand = np.random.permutation(dataTraining.shape[2])[0:int(0.5*dataTraining.shape[2])]
            dataTraining[:,nn,indrand] = dataTrainingNoNaN[:,nn,indrand]

        for nn in range(0,dataTraining.shape[1],time_step_obs):
            indrand = np.random.permutation(dataTraining.shape[2])[0:int(0.5*dataTraining.shape[2])]
            dataTest[:,nn,indrand] = dataTestNoNaN[:,nn,indrand]
            
        for nn in range(0,dataTraining.shape[1],time_step_obs):
            indrand = np.random.permutation(dataTraining.shape[2])[0:int(0.5*dataTraining.shape[2])]
            dataVal[:,nn,indrand] = dataValNoNaN[:,nn,indrand]


    genSuffixObs    = '_ObsSubRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
    print('... Data type: '+genSuffixObs)
                #for nn in range(0,dataTraining.shape[1],time_step_obs):
                #    dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
                
                #dataTest    = np.zeros((dataTestNoNaN.shape))
                #dataTest[:] = float('nan')
                #dataTest[:,::time_step_obs,:] = dataTestNoNaN[:,::time_step_obs,:]

print(dataTraining[0,40:60,:])    

            # set to NaN patch boundaries
if 1*0:
    dataTraining[:,0:10,:] =  float('nan')
    dataTest[:,0:10,:]     =  float('nan')
    dataVal[:,0:10,:] =  float('nan')
    dataTraining[:,dT-10:dT,:] =  float('nan')
    dataTest[:,dT-10:dT,:]     =  float('nan')
    dataVal[:,dT-10:dT,:]     =  float('nan')
    
            # mask for NaN
maskTraining = (dataTraining == dataTraining).astype('float')
maskVal     = ( dataVal    ==  dataVal ).astype('float')
maskTest     = ( dataTest    ==  dataTest   ).astype('float')
            
dataTraining = np.nan_to_num(dataTraining)
dataVal     = np.nan_to_num(dataVal)
dataTest     = np.nan_to_num(dataTest)
print(dataTraining[0,40:60,:]) 
print(np.mean(dataTraining))
# Permutation to have channel as #1 component
dataTraining      = np.moveaxis(dataTraining,-1,1)
maskTraining      = np.moveaxis(maskTraining,-1,1)
dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)

dataVal     = np.moveaxis(dataVal,-1,1)
maskVal      = np.moveaxis(maskVal,-1,1)
dataValNoNaN = np.moveaxis(dataValNoNaN,-1,1)
                        
dataTest      = np.moveaxis(dataTest,-1,1)
maskTest      = np.moveaxis(maskTest,-1,1)
dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)
            
    # set to NaN patch boundaries
    #dataTraining[:,0:5,:] =  dataTrainingNoNaN[:,0:5,:]
            #dataTest[:,0:5,:]     =  dataTestNoNaN[:,0:5,:]
            
############################################
## raw data
X_train         = dataTrainingNoNaN
X_train_missing = dataTraining
mask_train      = maskTraining

X_val         = dataValNoNaN
X_val_missing = dataVal
mask_val      = maskVal
                        
X_test         = dataTestNoNaN
X_test_missing = dataTest
mask_test      = maskTest
            
############################################
## normalized data
print(np.mean(mask_train))
meanTr          = np.mean(X_train_missing[:]) / np.mean(mask_train) 
print("mean_Val:")
print( np.mean(X_val_missing[:]) / np.mean(mask_val))
print("mean_Test:")
print( np.mean(X_test_missing[:]) / np.mean(mask_test))    

x_train_missing = X_train_missing - meanTr
x_val_missing = X_val_missing - meanTr
x_test_missing  = X_test_missing - meanTr

            
# scale wrt std
stdTr           = np.sqrt( np.mean( X_train_missing**2 ) / np.mean(mask_train) )
x_train_missing = x_train_missing / stdTr
x_val_missing  = x_val_missing / stdTr
x_test_missing  = x_test_missing / stdTr
            
x_train = (X_train - meanTr) / stdTr
x_val  = (X_val - meanTr) / stdTr
x_test  = (X_test - meanTr) / stdTr
print(meanTr)
print(stdTr)
print(np.mean(np.mean(x_train,0),1))
print(np.mean((x_train)**2))


        
# Generate noisy observsation
X_train_obs = X_train_missing + sigNoise * maskTraining * np.random.randn(X_train_missing.shape[0],X_train_missing.shape[1],X_train_missing.shape[2])
X_val_obs  = X_val_missing  + sigNoise * maskVal * np.random.randn(X_val_missing.shape[0],X_val_missing.shape[1],X_val_missing.shape[2])
X_test_obs  = X_test_missing  + sigNoise * maskTest * np.random.randn(X_test_missing.shape[0],X_test_missing.shape[1],X_test_missing.shape[2])
            
x_train_obs = (X_train_obs - meanTr) / stdTr
x_val_obs  = (X_val_obs - meanTr) / stdTr
x_test_obs  = (X_test_obs - meanTr) / stdTr

print('..... Training dataset: %dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
print('..... Validation dataset    : %dx%dx%d'%(x_val.shape[0],x_val.shape[1],x_val.shape[2]))
print('..... Test dataset    : %dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))


print('........ Initialize interpolated states')

# Initialization for interpolation
flagInit = 3
    
if flagInit == 0: 
    X_train_Init = mask_train * X_train_obs + (1. - mask_train) * (np.zeros(X_train_missing.shape) + meanTr)
    X_val_Init = mask_val * X_val_obs + (1. - mask_val) * (np.zeros(X_val_missing.shape) + meanTr)
    X_test_Init  = mask_test * X_test_obs + (1. - mask_test) * (np.zeros(X_test_missing.shape) + meanTr)
else:
    X_train_Init = np.zeros(X_train.shape)
    for ii in range(0,X_train.shape[0]):
    # Initial linear interpolation for each component
        XInit = np.zeros((X_train.shape[1],X_train.shape[2]))
            
        for kk in range(0,3):
            indt  = np.where( mask_train[ii,kk,:] == 1.0 )[0]
            indt_ = np.where( mask_train[ii,kk,:] == 0.0 )[0]
            
            if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_train_obs[ii,kk,indt])
                XInit[kk,indt]  = X_train_obs[ii,kk,indt]
                XInit[kk,indt_] = fkk(indt_)
            else:
                XInit = XInit + meanTr
            
        X_train_Init[ii,:,:] = XInit
        
    X_val_Init = np.zeros(X_val.shape)
    for ii in range(0,X_val.shape[0]):
    # Initial linear interpolation for each component
        XInit = np.zeros((X_val.shape[1],X_val.shape[2]))
            
        for kk in range(0,3):
            indt  = np.where( mask_val[ii,kk,:] == 1.0 )[0]
            indt_ = np.where( mask_val[ii,kk,:] == 0.0 )[0]
            
            if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_val_obs[ii,kk,indt])
                XInit[kk,indt]  = X_val_obs[ii,kk,indt]
                XInit[kk,indt_] = fkk(indt_)
            else:
                XInit = XInit + meanTr
            
        X_val_Init[ii,:,:] = XInit
            
    X_test_Init = np.zeros(X_test.shape)
    for ii in range(0,X_test.shape[0]):
    # Initial linear interpolation for each component
        XInit = np.zeros((X_test.shape[1],X_test.shape[2]))
            
        for kk in range(0,X_test.shape[1]):
            indt  = np.where( mask_test[ii,kk,:] == 1.0 )[0]
            indt_ = np.where( mask_test[ii,kk,:] == 0.0 )[0]
            
            if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_test_obs[ii,kk,indt])
                XInit[kk,indt]  = X_test_obs[ii,kk,indt]
                XInit[kk,indt_] = fkk(indt_)
            else:
                XInit = XInit + meanTr
            
        X_test_Init[ii,:,:] = XInit
                  #plt.figure()
                  #plt.figure()
                  #plt.plot(YObs[0:200,1],'r.')
                  #plt.plot(XGT[0:200,1],'b-')
                  #plt.plot(XInit[0:200,1],'k-')
                        
                        
x_train_Init = ( X_train_Init - meanTr ) / stdTr
x_val_Init = ( X_val_Init - meanTr ) / stdTr
x_test_Init = ( X_test_Init - meanTr ) / stdTr


if flagInit==3:#full covariance
    
    from netCDF4 import Dataset
    ncfile = Dataset('Results/Experiment5/train_set.nc')
    mu = ncfile['preds']
    prec = ncfile['p_diag']
    init   = ncfile['inits']
    target = ncfile['targets']
    masks = ncfile['masks']
    
    
    xt                 = np.zeros((X_train_Init.shape[0],9,X_train_Init.shape[2]))
    xt[:,3:6,:]        = np.abs(prec[:,:,:])
    xt[:,:3,:]         = init[:,:,:]
    x_train_Init       = xt
    mask_train         = masks[:,:,:]
    x_train            = target[:,:,:]
    
    ncfile = Dataset('Results/Experiment5/val_set.nc')
    mu = ncfile['preds']
    prec = ncfile['p_diag']
    init   = ncfile['inits']
    target = ncfile['targets']
    masks = ncfile['masks']
    
    xv                 = np.zeros((X_val_Init.shape[0],9,X_val_Init.shape[2]))
    xv[:,3:6,:]        = np.abs(prec[:,:,:])
    xv[:,:3,:]         = init[:,:,:]
    x_val_Init         = xv
    mask_val           = masks
    x_val              = target
    
    ncfile = Dataset('Results/Experiment5/test_set.nc')
    mu = ncfile['preds']
    prec = ncfile['p_diag']
    init   = ncfile['inits']
    target = ncfile['targets']
    masks = ncfile['masks']
   
    xtt                 = np.zeros((X_test_Init.shape[0],9,X_test_Init.shape[2]))
    xtt[:,3:6,:]        = np.abs(prec[:,:,:])
    xtt[:,:3,:]         = init[:,:,:]
    x_test_Init         = xtt
    mask_test           = masks
    x_test              = target

if flagInit==2:    
    Xt                 = np.zeros((X_train_Init.shape[0],6,X_train_Init.shape[2]))
    xt                 = np.zeros((X_train_Init.shape[0],6,X_train_Init.shape[2]))
    Xt[:,3:,:]        += 1
    xt[:,3:,:]        += 1
    Xt[:,:3,:]         = X_train_Init
    xt[:,:3,:]         = x_train_Init
    x_train_Init       = xt
    X_train_Init       = Xt

    Xv                 = np.zeros((X_val_Init.shape[0],6,X_val_Init.shape[2]))
    xv                 = np.zeros((X_val_Init.shape[0],6,X_val_Init.shape[2]))
    Xv[:,3:,:]        += 1
    xv[:,3:,:]        += 1    
    Xv[:,:3,:]         = X_val_Init
    xv[:,:3,:]         = x_val_Init
    x_val_Init         = xv
    X_val_Init         = Xv
           
    Xtt                 = np.zeros((X_test_Init.shape[0],6,X_test_Init.shape[2]))
    xtt                 = np.zeros((X_test_Init.shape[0],6,X_test_Init.shape[2]))
    Xtt[:,3:,:]        += 1
    xtt[:,3:,:]        += 1     
    Xtt[:,:3,:]         = X_test_Init
    xtt[:,:3,:]         = x_test_Init
    x_test_Init         = xtt
    X_test_Init         = Xtt
    
if flagInit > 1:     
    print('..... Training dataset: %dx%dx%d'%(x_train_Init.shape[0],x_train_Init.shape[1],x_train_Init.shape[2]))
    print('..... Validation dataset: %dx%dx%d'%(x_val_Init.shape[0],x_val_Init.shape[1],x_val_Init.shape[2]))
    print('..... Test dataset    : %dx%dx%d'%(x_test_Init.shape[0],x_test_Init.shape[1],x_test_Init.shape[2]))

else :
    print('..... Training dataset: %dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
    print('..... Validation dataset: %dx%dx%d'%(x_val.shape[0],x_val.shape[1],x_val.shape[2]))
    print('..... Test dataset    : %dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))


training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init),torch.Tensor(x_train_obs),torch.Tensor(mask_train),torch.Tensor(x_train)) # create your datset

val_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_val_Init),torch.Tensor(x_val_obs),torch.Tensor(mask_val),torch.Tensor(x_val)) # create your datset
     
test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_Init),torch.Tensor(x_test_obs),torch.Tensor(mask_test),torch.Tensor(x_test)) 

dataloaders = {
                'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
                'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
                'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            }            
dataset_sizes = {'train': len(training_dataset),'val': len(val_dataset), 'test': len(test_dataset)}

