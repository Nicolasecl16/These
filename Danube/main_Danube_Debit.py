#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:59:23 2020
@author: rfablet
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import xarray as xr
import argparse
import numpy as np
import datetime

import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from netCDF4 import Dataset

import os
import sys
sys.path.append('../4dvarnet-core')
import solver as NN_4DVar

from sklearn.feature_extraction import image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy import ndimage
#import torch.distributed as dist

## NN architectures and optimization parameters
batch_size      = 2#16#4#4#8#12#8#256#
DimAE           = 50#10#10#50
dimGradSolver   = 100 # dimension of the hidden state of the LSTM cell
rateDropout     = 0.25 # dropout rate 

# data generation
sigNoise        = 0. ## additive noise standard deviation
flagSWOTData    = True #False ## use SWOT data or not
flagNoSSTObs = True #False #
dT              = 5 ## Time window of each space-time patch
W               = 200 ## width/height of each space-time patch
dx              = 1   ## subsampling step if > 1
Nbpatches       = 1#10#10#25 ## number of patches extracted from each time-step 
rnd1            = 0 ## random seed for patch extraction (space sam)
rnd2            = 100 ## random seed for patch extraction
dwscale         = 1

# loss
p_norm_loss = 2. 
q_norm_loss = 2. 
r_norm_loss = 2. 
thr_norm_loss = 0.

W = int(W/dx)

UsePriodicBoundary = False # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
InterpFlag         = False # True => force reconstructed field to observed data after each gradient-based update

# Definiton of training, validation and test dataset
# from dayly indices over a one-year time series

#Se restreindre à l'été car pas de pluie??
day0=datetime.date(1960,1,1)
dayend=datetime.date(2009,12,12)

############################################## Data generation ###############################################################
print('........ Random seed set to 100')
np.random.seed(100)
torch.manual_seed(100)

genSuffixObs = ''


ncfile = Dataset('Dataset_Danube.nc',"r")
#qHR    = ncfile.variables['ssh'][:]
# select GF region
lon = ncfile.variables['lon'][:]
lat = ncfile.variables['lat'][:]
idLat = np.where((lat >= 33.) & (lat <= 43.))[0]
idLon = np.where((lon >= -65.) & (lon <= -55.))[0]
lon = lon[idLon]
lat = lat[idLat]
qHR    = ncfile.variables['ssh'][:, idLat, idLon]
ncfile.close()

ncfile = Dataset(dirDATA+'NATL60-CJM165_NATL_sst_y2013.1y.nc',"r")
qSST   = ncfile.variables['sst'][:, idLat, idLon]
ncfile.close()

ncfile = Dataset(dirDATA+"NATL60-CJM165_GULFSTREAM_u_y2013.1y.nc","r")
q_u = ncfile.variables['u'][:]
ncfile = Dataset(dirDATA+"NATL60-CJM165_GULFSTREAM_v_y2013.1y.nc","r")
q_v = ncfile.variables['v'][:]

if flagSWOTData == True :
    print('.... Use SWOT+4-nadir dataset')
    genFilename  = 'resInterpSLAwSWOT_Exp3_NewSolver_'+str('%03d'%(W))+'x'+str('%03d'%(W))+'x'+str('%02d'%(dT))
    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(dirDATA+"/ssh_NATL60_swot_4nadir.nc","r")
    qOI    = ncfile.variables['ssh_mod'][:, idLat, idLon]
    ncfile.close()

    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(dirDATA+"/dataset_nadir_0d_swot.nc","r")
    
    qMask   = ncfile.variables['ssh_mod'][:, idLat, idLon]
    qMask   = 1.0-qMask.mask.astype(float)
    ncfile.close()

else:
    genFilename  = 'resInterp4NadirSLAwOInoSST_'+str('%03d'%(W))+'x'+str('%03d'%(W))+'x'+str('%02d'%(dT))
    print('.... Use 4-nadir dataset')
    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(dirDATA+"oi/ssh_NATL60_4nadir.nc","r")
    qOI    = ncfile.variables['ssh_mod'][:]
    ncfile.close()

    # OI data using a noise-free OSSE (ssh_mod variable)
    ncfile = Dataset(dirDATA+"data/dataset_nadir_0d.nc","r")
    
    qMask   = ncfile.variables['ssh_mod'][:]
    qMask   = 1.0-qMask.mask.astype(float)
    ncfile.close()

print('----- MSE OI: %.3f'%np.mean((qOI-qHR)**2))
print()



## extraction of patches from the SSH field
#NoRndPatches = False  
#if ( Nbpatches == 1 ) & ( W == 200 ):
#NoRndPatches = True
print('... No random seed for the extraction of patches')

qHR   = qHR[:,0:200,0:200]
qOI   = qOI[:,0:200,0:200]
qMask = qMask[:,0:200,0:200]
qSST = qSST[:,0:200,0:200]
q_u = q_u[:,0:200,0:200]
q_v = q_v[:,0:200,0:200]
    
def extract_SpaceTimePatches(q,i1,i2,W,dT,rnd1,rnd2,D=1):
    dataTraining  = image.extract_patches_2d(np.moveaxis(q[i1:i2,::D,::D], 0, -1),(W,W),max_patches=Nbpatches,random_state=rnd1)
    dataTraining  = np.moveaxis(dataTraining, -1, 1)
    dataTraining  = dataTraining.reshape((Nbpatches,dataTraining.shape[1],W*W)) 
    
    dataTraining  = image.extract_patches_2d(dataTraining,(Nbpatches,dT),max_patches=None)

    dataTraining  = dataTraining.reshape((dataTraining.shape[0],dataTraining.shape[1],dT,W,W)) 
    dataTraining  = np.moveaxis(dataTraining, 0, -1)
    dataTraining  = np.moveaxis(dataTraining, 0, -1)
    dataTraining  = dataTraining.reshape((dT,W,W,dataTraining.shape[3]*dataTraining.shape[4])) 
    dataTraining  = np.moveaxis(dataTraining, -1, 0)
    return dataTraining     

# training dataset
dataTraining1     = extract_SpaceTimePatches(qHR,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
dataTrainingMask1 = extract_SpaceTimePatches(qMask,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
dataTrainingOI1   = extract_SpaceTimePatches(qOI,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
dataTrainingSST1 = extract_SpaceTimePatches(qSST,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
dataTraining_u1 = extract_SpaceTimePatches(q_u,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)
dataTraining_v1 = extract_SpaceTimePatches(q_v,iiTr1,jjTr1,W,dT,rnd1,rnd2,dx)

dataTraining2     = extract_SpaceTimePatches(qHR,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
dataTrainingMask2 = extract_SpaceTimePatches(qMask,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
dataTrainingOI2  = extract_SpaceTimePatches(qOI,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
dataTrainingSST2  = extract_SpaceTimePatches(qSST,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
dataTraining_u2 = extract_SpaceTimePatches(q_u,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)
dataTraining_v2 = extract_SpaceTimePatches(q_v,iiTr2,jjTr2,W,dT,rnd1,rnd2,dx)

dataTraining      = np.concatenate((dataTraining1,dataTraining2),axis=0)
dataTrainingMask  = np.concatenate((dataTrainingMask1,dataTrainingMask2),axis=0)
dataTrainingOI    = np.concatenate((dataTrainingOI1,dataTrainingOI2),axis=0)
dataTrainingSST    = np.concatenate((dataTrainingSST1,dataTrainingSST2),axis=0)
dataTraining_u    = np.concatenate((dataTraining_u1,dataTraining_u2),axis=0)
dataTraining_v    = np.concatenate((dataTraining_v1,dataTraining_v2),axis=0)

# test dataset
dataTest     = extract_SpaceTimePatches(qHR,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTestMask = extract_SpaceTimePatches(qMask,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTestOI   = extract_SpaceTimePatches(qOI,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTestSST  = extract_SpaceTimePatches(qSST,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTest_u  = extract_SpaceTimePatches(q_u,iiTest,jjTest,W,dT,rnd1,rnd2,dx)
dataTest_v  = extract_SpaceTimePatches(q_v,iiTest,jjTest,W,dT,rnd1,rnd2,dx)

# validation dataset
dataVal     = extract_SpaceTimePatches(qHR,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataValMask = extract_SpaceTimePatches(qMask,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataValOI   = extract_SpaceTimePatches(qOI,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataValSST  = extract_SpaceTimePatches(qSST,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataVal_u  = extract_SpaceTimePatches(q_u,iiVal,jjVal,W,dT,rnd1,rnd2,dx)
dataVal_v  = extract_SpaceTimePatches(q_v,iiVal,jjVal,W,dT,rnd1,rnd2,dx)

meanTr     = np.mean(dataTraining)
x_train    = dataTraining - meanTr
stdTr      = np.sqrt( np.mean( x_train**2 ) )
x_train    = x_train / stdTr

meanSST      = np.mean(dataTrainingSST)
ySST_train   = dataTrainingSST - meanSST
stdSST       = np.sqrt( np.mean( ySST_train**2 ) )
ySST_train   = ySST_train / stdSST

print('... mean current magntude: %.2e'%np.mean( np.sqrt(dataTraining_u**2 + dataTraining_v**2 ) ) )
std_uv = np.sqrt( np.mean( dataTraining_u**2 + dataTraining_v**2 ) )

u_train      = dataTraining_u / std_uv     
v_train      = dataTraining_v / std_uv     

x_trainOI   = (dataTrainingOI - meanTr) / stdTr
x_trainMask = dataTrainingMask

x_test     = (dataTest  - meanTr )
stdTt      = np.sqrt( np.mean( x_test**2 ) )
x_test     = x_test / stdTr
x_testOI   = (dataTestOI - meanTr) / stdTr
x_testMask  = dataTestMask
ySST_test  = (dataTestSST - meanSST ) / stdSST
u_test      = dataTest_u / std_uv     
v_test      = dataTest_v / std_uv     


x_val     = (dataVal  - meanTr )
stdVal    = np.sqrt( np.mean( x_val**2 ) )
x_val     = x_val / stdTr
x_valOI   = (dataValOI - meanTr) / stdTr
x_valMask = dataValMask
ySST_val  = (dataValSST - meanSST ) / stdSST
u_val      = dataVal_u / std_uv     
v_val      = dataVal_v / std_uv     

print('----- MSE Tr OI: %.6f'%np.mean((dataTrainingOI[:,int(dT/2),:,:]-dataTraining[:,int(dT/2),:,:])**2))
print('----- MSE Tt OI: %.6f'%np.mean((dataTestOI[:,int(dT/2),:,:]-dataTest[:,int(dT/2),:,:])**2))

print('..... Training dataset: %dx%dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
print('..... Test dataset    : %dx%dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

print('..... Masked points (Tr)) : %.3f'%(np.sum(x_trainMask)/(x_trainMask.shape[0]*x_trainMask.shape[1]*x_trainMask.shape[2]*x_trainMask.shape[3])))
print('..... Masked points (Tt)) : %.3f'%(np.sum(x_testMask)/(x_testMask.shape[0]*x_testMask.shape[1]*x_testMask.shape[2]*x_testMask.shape[3])) )

print('----- MSE Tr OI: %.6f'%np.mean(stdTr**2 * (x_trainOI[:,int(dT/2),:,:]-x_train[:,int(dT/2),:,:])**2))
print('----- MSE Tt OI: %.6f'%np.mean(stdTr**2 * (x_testOI[:,int(dT/2),:,:]-x_test[:,int(dT/2),:,:])**2))

######################### data loaders
training_dataset   = torch.utils.data.TensorDataset(torch.Tensor(x_trainOI),torch.Tensor(x_trainMask),torch.Tensor(ySST_train),torch.Tensor(x_train),torch.Tensor(u_train),torch.Tensor(v_train)) # create your datset
val_dataset        = torch.utils.data.TensorDataset(torch.Tensor(x_valOI),torch.Tensor(x_valMask),torch.Tensor(ySST_val),torch.Tensor(x_val),torch.Tensor(u_val),torch.Tensor(v_val)) # create your datset
test_dataset       = torch.utils.data.TensorDataset(torch.Tensor(x_testOI),torch.Tensor(x_testMask),torch.Tensor(ySST_test),torch.Tensor(x_test),torch.Tensor(u_test),torch.Tensor(v_test))  # create your datset

dataloaders = {
    'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
}            

var_Tr    = np.var( x_train )
var_Tt    = np.var( x_test )
var_Val   = np.var( x_val )

#######################################Phi_r, Model_H, Model_Sampling architectures ################################################

print('........ Define AE architecture')
shapeData      = np.array(x_train.shape[1:])
shapeData_test = np.array(x_test.shape[1:])
shapeData[0]  += 3*shapeData[0]
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

class Model_H(torch.nn.Module):
    def __init__(self,dim=5):
        super(Model_H, self).__init__()

        self.DimObs        = 2
        self.dimObsChannel = np.array([shapeData[0],dim])

        self.conv11  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(3,3),padding=1,bias=False)
        self.conv12  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(3,3),padding=1,bias=False)
                    
        self.conv21  = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(3,3),padding=1,bias=False)
        self.conv22  = torch.nn.Conv2d(shapeData[0],self.dimObsChannel[1],(3,3),padding=1,bias=False)
        self.convM   = torch.nn.Conv2d(shapeDataSST[0],self.dimObsChannel[1],(3,3),padding=1,bias=False)
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

def save_NetCDF(saved_path1, ind_start,ind_end, ssh_gt , ssh_oi, ssh_obs, sst_gt, u_gt, v_gt, ssh_rec, u_rec_geo, v_rec_geo, u_rec, v_rec, feat_sst ):
    
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
                   'ssh_rec': (('time', 'lat', 'lon'), ssh_rec), \
                   'u_rec_geo': (('time', 'lat_uv', 'lon_uv'), u_rec_geo), \
                   'v_rec_geo': (('time', 'lat_uv', 'lon_uv'), v_rec_geo), \
                   'u_rec': (('time', 'lat_uv', 'lon_uv'), u_rec), \
                   'v_rec': (('time', 'lat_uv', 'lon_uv'), v_rec), \
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

    for ssh_OI, inputs_Mask, inputs_SST, ssh_gt, u_gt, v_gt in dataloaders['train']:
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
        self.alpha_mse_uv = 1.

        self.p_norm_loss = 2. 
        self.q_norm_loss = 2. 
        self.r_norm_loss = 2. 
        self.thr_norm_loss = 2.
        
        self.alpha_ssh2u = 1.
        self.alpha_ssh2v = 1.

        self.flagNoSSTObs = False
        
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
        self.hparams.alpha_mse_uv = 1.
        self.hparams.alpha_mse_div = 1.
        self.hparams.alpha_mse_curl = 1.
        self.hparams.flag_uv_param = "u-v"
        
        self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(w_), requires_grad=False)
        self.hparams.automatic_optimization = False#True#

        self.hparams.alpha_ssh2u = alpha_ssh2u
        self.hparams.alpha_ssh2v = alpha_ssh2v
        self.hparams.d_du = 1.
        self.hparams.d_dv = alpha_ssh2u / alpha_ssh2v

        # main model
        self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                         Model_H(), 
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
        self.log("test_uv", metrics['mse_uv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
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

        x_test_ssh_obs = torch.cat([chunk['obs_ssh'] for chunk in outputs]).numpy()
        x_test_ssh_obs[ x_test_ssh_obs == 0. ] = np.float('NaN')
        x_test_ssh_obs = stdTr * x_test_ssh_obs + meanTr
        self.x_rec_ssh_obs = x_test_ssh_obs[:,int(dT/2),:,:]

        x_test_sst_feat = torch.cat([chunk['feat_sst'] for chunk in outputs]).numpy()
        self.x_feat_sst = x_test_sst_feat

        return 1.

    def compute_loss(self, batch, phase):

        ssh_OI, inputs_Mask, inputs_SST, ssh_GT, u_GT, v_GT = batch

        new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask, 0. * inputs_Mask , 0. * inputs_Mask ), dim=1)
        inputs_init    = torch.cat((ssh_OI, inputs_Mask * (ssh_GT - ssh_OI) , 0. * ssh_GT , 0. * ssh_GT ), dim=1)
        inputs_missing = torch.cat((ssh_OI, inputs_Mask * (ssh_GT - ssh_OI) , 0. * ssh_GT , 0. * ssh_GT), dim=1)
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


class LitModel_Normpq(LitModel):
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
        self.hparams.alpha_mse_uv = 1.
        self.hparams.alpha_mse_div = 0.#100. * 1.e2
        self.hparams.alpha_mse_curl = 1.e1
        self.hparams.flag_uv_param = "u-v"
        
        if self.hparams.alpha_mse_uv == 0. :
            self.hparams.alpha_mse_div = 0.
            self.hparams.alpha_mse_curl = 0.
            
        self.hparams.p_norm_loss = p_norm_loss 
        self.hparams.q_norm_loss = q_norm_loss
        self.hparams.r_norm_loss = r_norm_loss 
        self.hparams.thr_norm_loss = thr_norm_loss
        self.hparams.thr_div = 0.#5e-2
        
        # SSH gradient to current
        self.hparams.alpha_ssh2u = alpha_ssh2u
        self.hparams.alpha_ssh2v = alpha_ssh2v
        self.hparams.d_du = 1.
        self.hparams.d_dv = alpha_ssh2u / alpha_ssh2v
        
        self.hparams.flagNoSSTObs = flagNoSSTObs
        
        self.hparams.w_loss  = torch.nn.Parameter(torch.Tensor(w_), requires_grad=False)
        self.hparams.automatic_optimization = False#True#

        # main model
        self.model        = NN_4DVar.Solver_Grad_4DVarNN(Phi_r(), 
                                                         Model_H(), 
                                                         NN_4DVar.model_GradUpdateLSTM(shapeData, UsePriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout), 
                                                         NN_4DVar.Model_WeightedL2Norm(), Model_NomrLpqr(2.,2.,2.,True), shapeData, self.hparams.n_grad)

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

    def compute_divcurl(self,u,v):
        du_du,du_dv = self.compute_graduv(u) 
        dv_du,dv_dv = self.compute_graduv(v) 
                
        du_du *= self.hparams.d_du
        du_dv *= self.hparams.d_dv
        dv_du *= self.hparams.d_du
        dv_dv *= self.hparams.d_dv

        div = du_du + dv_dv 
        curl = du_dv - dv_du
        
        return div,curl

    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')

        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse'] / var_Val , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_uv", metrics['mse_uv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_div", metrics['mse_div'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics = self.compute_loss(test_batch, phase='test')

        self.log('test_loss', loss)
        self.log("test_mse", metrics['mse'] / var_Tt , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_div", metrics['mse_div'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("test_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_uv", metrics['mse_uv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        out_ssh,out_u_geo,out_v_geo,out_u,out_v, ssh_obs,sst_feat = out
        
        return {'preds_ssh': out_ssh.detach().cpu(),'preds_u_geo': out_u_geo.detach().cpu(),'preds_v_geo': out_v_geo.detach().cpu(),'preds_u': out_u.detach().cpu(),'preds_v': out_v.detach().cpu(),'obs_ssh': ssh_obs.detach().cpu(),'feat_sst': sst_feat.detach().cpu()}
    def compute_loss(self, batch, phase):

        ssh_OI, inputs_Mask, inputs_SST, ssh_GT, u_GT, v_GT = batch

        new_masks      = torch.cat((1. + 0. * inputs_Mask, inputs_Mask, 0. * inputs_Mask , 0. * inputs_Mask ), dim=1)
        inputs_init    = torch.cat((ssh_OI, inputs_Mask * (ssh_GT - ssh_OI) , 0. * ssh_GT , 0. * ssh_GT ), dim=1)
        inputs_missing = torch.cat((ssh_OI, inputs_Mask * (ssh_GT - ssh_OI) , 0. * ssh_GT , 0. * ssh_GT), dim=1)
        mask_SST       = 1. + 0. * inputs_SST

        if self.hparams.flagNoSSTObs == True :
            mask_SST = 0. * mask_SST
            inputs_SST = 0. * inputs_SST
            
        # gradient norm field
        #gx_targets_GT, gy_targets_GT, ng_targets_GT  = self.gradient_img(ssh_GT)
        gu_targets_GT, gv_targets_GT  = self.compute_graduv(ssh_GT)
        ng_targets_GT = torch.sqrt( gu_targets_GT**2 + gv_targets_GT**2 )
        
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
            gu_outputs, gv_outputs  = self.compute_graduv(outputs_ssh)
            loss_ssh   = NN_4DVar.compute_WeightedLoss((outputs_ssh - ssh_GT), self.w_loss)
            d_gssh = torch.sqrt( 1e-10 + ( gu_outputs - gu_targets_GT) ** 2 + ( gv_outputs - gv_targets_GT) ** 2 )
            loss_gssh  = NN_4DVar.compute_WeightedLoss( d_gssh, self.w_loss)

            gu_oi, gv_oi  = self.compute_graduv(ssh_OI)
            d_goi = torch.sqrt( 1e-10 + ( gu_oi - gu_targets_GT) ** 2 + ( gv_oi - gv_targets_GT) ** 2 )
            loss_goi   = NN_4DVar.compute_WeightedLoss(d_goi, self.w_loss)

            loss_oi    = NN_4DVar.compute_WeightedLoss(ssh_GT - ssh_OI, self.w_loss)
            

            # reconstruction losses for current 
            if self.hparams.flag_uv_param =='div-curl':            
                dssh_u,dssh_v = self.compute_graduv(outputs_ssh)     
                outputs_u_geo = -1. * self.hparams.alpha_ssh2u * dssh_v
                outputs_v_geo = 1. * self.hparams.alpha_ssh2v * dssh_u              

                dssh_u,dssh_v = self.compute_graduv(outputs_uv1)
                outputs_u_geo += -1. * self.hparams.alpha_ssh2u * dssh_v
                outputs_v_geo += 1. * self.hparams.alpha_ssh2v * dssh_u              

                dssh_u,dssh_v = self.compute_graduv(outputs_uv2)
                outputs_u = outputs_u_geo + self.hparams.alpha_ssh2v * dssh_u
                outputs_v = outputs_v_geo + self.hparams.alpha_ssh2u * dssh_v               

            else:
                outputs_u = outputs_uv1
                outputs_v = outputs_uv2

                dssh_u,dssh_v = self.compute_graduv(outputs_ssh)
     
                outputs_u_geo = -1. * self.hparams.alpha_ssh2u * dssh_v
                outputs_v_geo = 1. * self.hparams.alpha_ssh2v * dssh_u              
                
            # loss (u,v)
            alpha = 0.5
            alpha_r = alpha ** self.hparams.r_norm_loss
            d_uv = torch.sqrt( 1e-10 + (outputs_u - u_GT)**2 + (outputs_v - v_GT) **2 )
            if  self.hparams.thr_norm_loss == 0. :
                loss_uv = alpha_r * compute_WeightedLoss_Lpqr(d_uv / alpha, self.w_loss,self.hparams.p_norm_loss,self.hparams.q_norm_loss,self.hparams.r_norm_loss,self.hparams.thr_norm_loss)
            else:
                n_gt = torch.sigmoid( 10. *( u_GT**2 + v_GT **2 -  self.hparams.thr_norm_loss ** 2 )  )
                loss_uv = alpha_r * compute_WeightedLoss_Lpqr( n_gt * d_uv / alpha, self.w_loss,self.hparams.p_norm_loss,self.hparams.q_norm_loss,self.hparams.r_norm_loss,self.hparams.thr_norm_loss)
                    
                                    
            # loss div/curl    
            div_rec,curl_rec = self.compute_divcurl(outputs_u,outputs_v)
            div_gt,curl_gt = self.compute_divcurl(u_GT,v_GT)
                             
            if self.hparams.thr_div == 0. :
                loss_div = (div_rec - div_gt)
            else:                
                loss_div = torch.sigmoid( 100. * (torch.abs(div_gt) - self.hparams.thr_div) ) * (div_rec - div_gt)
            loss_div = NN_4DVar.compute_WeightedLoss(loss_div, self.w_loss)
            loss_curl = (curl_rec - curl_gt)
            loss_curl = NN_4DVar.compute_WeightedLoss(loss_curl, self.w_loss)

            outputs_u = outputs_u[:,:,1:-1,1:-1]
            outputs_v = outputs_v[:,:,1:-1,1:-1]
            
            if self.current_epoch == 1:
                self.curr = 1
                
            if self.curr == self.current_epoch :
                p_,q_,r_ = self.model.model_VarCost.normPrior.compute_pqr()
                print('... Prior norm: p= %f q=%f r=%f'%(p_,q_,r_))
                #print(' %e %e'%(torch.mean( torch.abs(div_gt)),torch.mean( torch.abs(div_rec))))
                #print(' %e %e'%(torch.mean( torch.abs(curl_gt)),torch.mean( torch.abs(curl_rec))))
                print(' %e %e'%(loss_div,loss_curl))
                print(' %% point > thr_div %.3f'%np.mean( np.abs(div_gt.detach().cpu().numpy()) > self.hparams.thr_div )  )
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
            loss = self.hparams.alpha_mse_ssh * loss_ssh + self.hparams.alpha_mse_gssh * loss_gssh 
            loss += self.hparams.alpha_mse_uv * loss_uv 
            loss += self.hparams.alpha_mse_div * loss_div + self.hparams.alpha_mse_curl * loss_curl
            loss += 0.5 * self.hparams.alpha_proj * (loss_AE + loss_AE_GT)
            loss    += self.hparams.alpha_lr * loss_LR + self.hparams.alpha_sr * loss_SR
            
            # metrics
            mean_gall = NN_4DVar.compute_WeightedLoss(ng_targets_GT,self.w_loss)
            mse = loss_ssh.detach()
            mse_grad   = loss_gssh.detach()
            mse_uv = loss_uv.detach()
            mse_div = loss_div.detach()
            metrics   = dict([('mse',mse),('mse_grad',mse_grad),('mse_div',mse_div),('mse_uv',mse_uv),('meanGrad',mean_gall),('mseOI',loss_oi.detach()),('mseGOI',loss_goi.detach())])
            #print(mse.cpu().detach().numpy())
            
            
            outputs_ssh_obs = inputs_missing[:,dT:2*dT,:,:] + new_masks[:,dT:2*dT,:,:] * inputs_missing[:,0:dT,:,:]
            outputs_obs_sst = self.model.model_H.conv21( inputs_SST )
            
            outputs = [outputs_ssh,outputs_u_geo,outputs_v_geo,outputs_u,outputs_v,outputs_ssh_obs,outputs_obs_sst]
        return loss,outputs, metrics
 

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

if __name__ == '__main__':
    
    flagProcess = 1
    
    if flagProcess == 0: ## training model from scratch
    
        loadTrainedModel = True#False#
        if loadTrainedModel == True :             
            
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-GF-Exp3-epoch=146-val_loss=0.10.ckpt'

            print('.... load pre-trained model :'+pathCheckPOint)
            mod = LitModel.load_from_checkpoint(pathCheckPOint)
            
            mod.hparams.n_grad          = 15
            mod.hparams.iter_update     = [0, 50, 100, 100, 150, 150, 800]  # [0,2,4,6,9,15]
            mod.hparams.nb_grad_update  = [15, 15, 15, 15, 15, 20, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            mod.hparams.lr_update       = [1e-4, 1e-5, 1e-6, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]

        else:
            mod = LitModel()
            mod2 = LitModel_Normpq()
        
        print(mod.hparams)
        print('n_grad = %d'%mod.model.n_grad)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath= dirSAVE+'-'+suffix_exp,
                                              filename='modelSSCurrent-uv-GF-Exp3-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=3,
                                              mode='min')
        profiler_kwargs = {'max_epochs': 200 }

        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1, **profiler_kwargs,callbacks=[checkpoint_callback])
    
        ## training loop
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
        X_val    = qHR[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_OI     = qOI[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
                
        val_mseRec = compute_metrics(X_val,mod.x_rec)     
        val_mseOI  = compute_metrics(X_val,X_OI)     
        
        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        #ncfile = Dataset("results/test.nc","r")
        #X_rec  = ncfile.variables['ssh'][:]
        #ncfile.close()
        X_test = qHR[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_OI   = qOI[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
    
            
        test_mseRec = compute_metrics(X_test,mod.x_rec)     
        test_mseOI  = compute_metrics(X_test,X_OI)     
        
        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(val_mseOI['mse'],val_mseRec['mse'],100. * (1.-val_mseRec['mse']/val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['mseGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['mseGrad'])))
    
    if flagProcess == 1: ## training model from scratch
    
        FLAG_TRAINABLE_NORM = False#True#
        
        loadTrainedModel = 0#False#
        if loadTrainedModel == 1 :             
            
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-GF-Exp3-epoch=146-val_loss=0.10.ckpt'

            print('.... load pre-trained model :'+pathCheckPOint)
            #mod = LitModel_Normpq.load_from_checkpoint(pathCheckPOint)
            mod2 = LitModel.load_from_checkpoint(pathCheckPOint)
            
            mod = LitModel_Normpq()

            mod.model.model_Grad = mod2.model.model_Grad
            mod.model.model_H = mod2.model.model_H
            mod.model.phi_r = mod2.model.phi_r
            mod.model_pot2curr = mod2.model_pot2curr
            mod.model.model_VarCost.alphaObs    = mod.model.model_VarCost.alphaObs
            mod.model.model_VarCost.alphaReg    = mod.model.model_VarCost.alphaReg
            mod.model.model_VarCost.WObs           = mod.model.model_VarCost.WObs
            mod.model.model_VarCost.WReg    = mod.model.model_VarCost.WReg
            mod.model.model_VarCost.epsObs = mod.model.model_VarCost.epsObs
            mod.model.model_VarCost.epsReg = mod.model.model_VarCost.epsReg
            
            mod.hparams.n_grad          = 5
            mod.hparams.iter_update     = [0, 20, 40, 60, 100, 150, 800]  # [0,2,4,6,9,15]
            mod.hparams.nb_grad_update  = [5, 10, 10, 15, 15, 20, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            mod.hparams.lr_update       = [1e-3, 1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 1e-6, 1e-7]

        elif  loadTrainedModel == 2 :      
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-Lpq-GF-exp3-Grad100-L2.0_2.0_2.0d-epoch=40-val_loss=0.10.ckpt'
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-exp3-Grad100-L2.0_2.0_2.0d-epoch=100-val_loss=0.10.ckpt'

            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-nodiv-exp3-Grad100-L2.0_2.0_2.0_0.0-epoch=70-val_loss=0.05.ckpt'            
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-div-exp3-Grad100-L2.0_2.0_2.0_0.0-epoch=107-val_loss=0.22.ckpt'
            
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-div-noSSTexp3-Grad100-L2.0_2.0_2.0_0.0-epoch=68-val_loss=0.40.ckpt'
            
            print('.... load pre-trained model :'+pathCheckPOint)
            mod = LitModel_Normpq.load_from_checkpoint(pathCheckPOint)
            #mod.compute_graduv = Compute_graduv()
 
            mod.hparams.n_grad          = 15
            mod.hparams.iter_update     = [0, 15, 50, 100, 150, 150, 800]  # [0,2,4,6,9,15]
            mod.hparams.nb_grad_update  = [15, 15, 15, 15, 15, 20, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
            mod.hparams.lr_update       = [1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-7]
        else:
            mod = LitModel_Normpq()
        
        if FLAG_TRAINABLE_NORM == True :
            filename_chkpt = 'modelSSCurrent-uv-LpqFromL22-GF-'
            #filename_chkpt = 'modelSSCurrent-uv-Lpq-GF-'
        else:
            filename_chkpt = 'modelSSCurrent-uv-L22-GF-'
            
        if mod.hparams.alpha_mse_uv == 0. :
            filename_chkpt = filename_chkpt + 'nouv-'            

        if mod.hparams.alpha_mse_div == 0. :
            filename_chkpt = filename_chkpt + 'nodiv-'
        else:
            filename_chkpt = filename_chkpt + 'div-'

        if mod.hparams.thr_div > 0. :
            filename_chkpt = filename_chkpt + 'thrdiv%.3f-'%mod.hparams.thr_div

        if mod.hparams.flagNoSSTObs == True :
            filename_chkpt = filename_chkpt + 'noSST-'
                   
        filename_chkpt = filename_chkpt + suffix_exp
        filename_chkpt = filename_chkpt+'-Grad%d'%dimGradSolver
        filename_chkpt = filename_chkpt+'-L%.1f_%.1f_%.1f_%.1f'%(mod.hparams.p_norm_loss,mod.hparams.q_norm_loss,mod.hparams.r_norm_loss,mod.hparams.thr_norm_loss)
        
        print('..... Filename chkpt: '+filename_chkpt)
        
        print(mod.hparams)
        print('n_grad = %d'%mod.hparams.n_grad)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath= dirSAVE+'-'+suffix_exp,
                                              filename= filename_chkpt + '-{epoch:02d}-{val_loss:.2f}',
                                              save_top_k=3,
                                              mode='min')
        profiler_kwargs = {'max_epochs': 200 }

        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1, **profiler_kwargs,callbacks=[checkpoint_callback])
    
        ## training loop
        trainer.fit(mod, dataloaders['train'], dataloaders['val'])
        
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
        X_val    = qHR[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_OI     = qOI[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
                
        val_mseRec = compute_metrics(X_val,mod.x_rec)     
        val_mseOI  = compute_metrics(X_val,X_OI)     
        
        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        #ncfile = Dataset("results/test.nc","r")
        #X_rec  = ncfile.variables['ssh'][:]
        #ncfile.close()
        X_test = qHR[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_OI   = qOI[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
    
            
        test_mseRec = compute_metrics(X_test,mod.x_rec)     
        test_mseOI  = compute_metrics(X_test,X_OI)     
        
        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(val_mseOI['mse'],val_mseRec['mse'],100. * (1.-val_mseRec['mse']/val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['mseGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['mseGrad'])))

        
    elif flagProcess == 2: ## test trained model with the non-Lighning code
        mod = LitModel()
        fileAEModelInit = './ResSLANATL60/resInterpSLAwSWOT_Exp3_NewSolver_200x200x05_GENN_2_50_03_01_01_25_HRObs_OIObs_MS_Grad_01_02_10_100_modelPHI_iter080.mod'
        
        mod.model.phi_r.load_state_dict(torch.load(fileAEModelInit))
        mod.model.model_Grad.load_state_dict(torch.load(fileAEModelInit.replace('_modelPHI_iter','_modelGrad_iter')))
        mod.model.model_VarCost.load_state_dict(torch.load(fileAEModelInit.replace('_modelPHI_iter','_modelVarCost_iter')))
        mod.model.NGrad = 10
    
        profiler_kwargs = {'max_epochs': 200}
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
        #ncfile = Dataset("results/test.nc","r")
        #X_rec    = ncfile.variables['ssh'][:]
        #ncfile.close()
        X_val    = qHR[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_OI     = qOI[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
                
        val_mseRec = compute_metrics(X_val,mod.x_rec)     
        val_mseOI  = compute_metrics(X_val,X_OI)     
        
        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        #ncfile = Dataset("results/test.nc","r")
        #X_rec  = ncfile.variables['ssh'][:]
        #ncfile.close()
        X_test = qHR[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_OI   = qOI[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
                
        test_mseRec = compute_metrics(X_test,mod.x_rec)     
        test_mseOI  = compute_metrics(X_test,X_OI)     
        
        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(val_mseOI['mse'],val_mseRec['mse'],100. * (1.-val_mseRec['mse']/val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['mseGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad'])))
        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['mseGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad'])))

    elif flagProcess == 3: ## test trained model with the Lightning code

        if 1*0 :
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-GF-Exp3-epoch=146-val_loss=0.10.ckpt'
            mod = LitModel.load_from_checkpoint(pathCheckPOint)     
        else:
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-exp3-Grad100-L2.0_2.0_2.0d-epoch=61-val_loss=0.09.ckpt'
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-divTEST-exp3-Grad100-L2.0_2.0_2.0_0.0-epoch=92-val_loss=0.05.ckpt'
            
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-div-exp3-Grad100-L2.0_2.0_2.0_0.0-epoch=107-val_loss=0.22.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-nodiv-exp3-Grad100-L2.0_2.0_2.0_0.0-epoch=149-val_loss=0.05.ckpt'
            
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-div-noSST-exp3-Grad100-L2.0_2.0_2.0_0.0-epoch=77-val_loss=0.40.ckpt'
            
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-LpqFromL22-GF-exp3-Grad100-L2.0_2.0_2.0d_2.0-epoch=55-val_loss=0.07.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-LpqFromL22-v2-GF-exp3-Grad100-L2.0_2.0_2.0_2.0-epoch=78-val_loss=0.07.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-LpqFromL22-GF-exp3-Grad100-L2.0_2.0_2.0_2.0-epoch=78-val_loss=0.07.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-exp3-Grad100-L2.0_2.0_2.0_2.0-epoch=79-val_loss=0.07.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-noPhiTr-exp3-Grad100-L2.0_2.0_2.0_2.0-epoch=18-val_loss=0.07.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-LpqFromL22-v2-GF-exp3-Grad100-L2.0_2.0_2.0_2.0-epoch=61-val_loss=0.07.ckpt'
                        
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-LpqFromL22-GF-exp3-Grad100-L2.0_2.0_2.0_1.0-epoch=19-val_loss=0.07.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-noPhiTr-exp3-Grad100-L2.0_2.0_2.0_1.0-epoch=68-val_loss=0.07.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-Lpq-GF-exp3-Grad100-L2.0_2.0_2.0d-epoch=56-val_loss=0.10.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-LpqFromL22-GF-exp3-Grad100-L2.0_2.0_2.0d-epoch=67-val_loss=0.09.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-LpqFromL22-GF-exp3-Grad100-L2.0_2.0_2.0d-epoch=90-val_loss=0.09.ckpt'
            
            #pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-divTEST-noSSTexp3-Grad100-L2.0_2.0_2.0_0.0-epoch=16-val_loss=0.49.ckpt'
            
            
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-div-exp5-Grad100-L2.0_2.0_2.0_0.0-epoch=59-val_loss=8.38.ckpt'
            pathCheckPOint = 'SSCurrNATL60_ChckPt/modelSSCurrent-uv-L22-GF-div-noSST-exp5-Grad100-L2.0_2.0_2.0_0.0-epoch=42-val_loss=13.73.ckpt'
            
            pathCheckPOint = 'SSCurrNATL60_ChckPt-winter/modelSSCurrent-uv-L22-GF-nodiv-winter-Grad100-L2.0_2.0_2.0_0.0-epoch=126-val_loss=0.23.ckpt'
            pathCheckPOint = 'SSCurrNATL60_ChckPt/-spring/modelSSCurrent-uv-L22-GF-nodiv-spring-Grad100-L2.0_2.0_2.0_0.0-epoch=127-val_loss=0.34.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt-fall/modelSSCurrent-uv-L22-GF-nodiv-fall-Grad100-L2.0_2.0_2.0_0.0-epoch=31-val_loss=0.14.ckpt'
            #pathCheckPOint = 'SSCurrNATL60_ChckPt-summer/modelSSCurrent-uv-L22-GF-nodiv-summer-Grad100-L2.0_2.0_2.0_0.0-epoch=123-val_loss=0.41.ckpt'
            
            mod = LitModel_Normpq.load_from_checkpoint(pathCheckPOint)
            
            p_,q_,r_ = mod.model.model_VarCost.normPrior.compute_pqr()
            print('... Trainable prior norm: p= %f q=%f r=%f'%(p_,q_,r_))
            
        print(mod.hparams)
        mod.hparams.n_grad = 15
        print(' Ngrad = %d / %d'%(mod.hparams.n_grad,mod.model.n_grad))
        profiler_kwargs = {'max_epochs': 200}
        #trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)
        trainer = pl.Trainer(gpus=1,  **profiler_kwargs)
        trainer.test(mod, test_dataloaders=dataloaders['val'])
        
        X_val    = qHR[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_OI     = qOI[iiVal+int(dT/2):jjVal-int(dT/2),:,:]
        X_u    = q_u[iiVal+int(dT/2):jjVal-int(dT/2),1:-1,1:-1]
        X_v    = q_v[iiVal+int(dT/2):jjVal-int(dT/2),1:-1,1:-1]
                
        var_val_uv = np.var( X_u ) + np.var( X_v )

        val_mseRec = compute_metrics(X_val,mod.x_rec_ssh)     
        val_mseOI  = compute_metrics(X_val,X_OI)     
        val_mse_u  = compute_metrics(X_u,mod.x_rec_u)     
        val_mse_v  = compute_metrics(X_v,mod.x_rec_v)     
        
        val_l_pqr = compute_Lpqr_numpy( [X_u,X_v] , [mod.x_rec_u,mod.x_rec_v],p_norm_loss,q_norm_loss,r_norm_loss,thr_norm_loss)
        
        print('\n\n........................................ ')
        print('........................................\n ')
        trainer.test(mod, test_dataloaders=dataloaders['test'])
        X_test = qHR[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_OI   = qOI[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        X_u    = q_u[iiTest+int(dT/2):jjTest-int(dT/2),1:q_u.shape[1]-1,1:q_u.shape[2]-1]
        X_v    = q_v[iiTest+int(dT/2):jjTest-int(dT/2),1:q_u.shape[1]-1,1:q_u.shape[2]-1]
         
        var_test_uv = np.var( X_u ) + np.var( X_v )
        
        test_mseRec = compute_metrics(X_test,mod.x_rec_ssh)     
        test_mseOI  = compute_metrics(X_test,X_OI)     
        test_mse_u  = compute_metrics(X_u,mod.x_rec_u)     
        test_mse_v  = compute_metrics(X_v,mod.x_rec_v)     

        # dvergence
        div_uv = ndimage.sobel(X_u,axis=0) + ndimage.sobel(X_v,axis=1)
        div_uv_rec = ndimage.sobel(mod.x_rec_u,axis=0) + ndimage.sobel(mod.x_rec_v,axis=1)
        mse_div = compute_metrics(div_uv,div_uv_rec)
        mse_l_pqr_div = compute_metrics(div_uv,div_uv_rec)
        
        test_l_pqr = compute_Lpqr_numpy( [X_u,X_v] , [mod.x_rec_u,mod.x_rec_v] , p_norm_loss,q_norm_loss,r_norm_loss,thr_norm_loss)
        test_var_l_pqr = compute_Lpqr_numpy( [X_u,X_v] , [0. * mod.x_rec_u,0. * mod.x_rec_v] , p_norm_loss,q_norm_loss,r_norm_loss,thr_norm_loss)
        
        saveRes = True #False# 
        ssh_gt = X_test
        ssh_oi = X_OI
        ssh_obs = mod.x_rec_ssh_obs
        u_gt = X_u
        v_gt = X_v
        sst_gt = qSST[iiTest+int(dT/2):jjTest-int(dT/2),:,:]
        u_rec = mod.x_rec_u
        v_rec = mod.x_rec_v
        ssh_rec = mod.x_rec_ssh            
        u_rec_geo = mod.x_rec_u_geo
        v_rec_geo = mod.x_rec_v_geo
        feat_sst = mod.x_feat_sst
        if saveRes == True :
            
            filename_res = pathCheckPOint.replace('.ckpt','_res.nc')
            print('.... save all gt/rec fields in nc file '+filename_res)
            save_NetCDF(filename_res, 
                        ind_start=iiTest+int(dT/2),
                        ind_end=jjTest-int(dT/2),
                        ssh_gt = ssh_gt , 
                        ssh_oi = ssh_oi, 
                        sst_gt = sst_gt,
                        ssh_obs = ssh_obs,
                        u_gt = u_gt,
                        v_gt = v_gt,
                        ssh_rec = ssh_rec,
                        u_rec_geo = u_rec_geo,
                        v_rec_geo = v_rec_geo,
                        u_rec = u_rec,
                        v_rec = v_rec,
                        feat_sst = feat_sst )

        print('...  model: '+pathCheckPOint)
        print('... Dataset: '+suffix_exp)
        print('... Evalaution norm: p= %f q=%f r=%f thr=%f'%(p_norm_loss,q_norm_loss,r_norm_loss,thr_norm_loss))
        print(' ')
        print('....................................')
        print('....... Validation dataset')
        print('....... MSE Val dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(val_mseOI['mse'],val_mseRec['mse'],100. * (1.-val_mseRec['mse']/val_mseOI['mse'])))
        print('....... MSE Val dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(val_mseOI['mseGrad'],val_mseRec['mseGrad'],100. * (1.-val_mseRec['mseGrad']/val_mseOI['meanGrad']),100. * (1.-val_mseRec['mseGrad']/val_mseOI['mseGrad'])))
        print('....... MSE Val dataset (u,v): %.3e / %.2f %%/ %.2f %%'%(val_mse_u['mse']+val_mse_v['mse'],100. * (1.-(val_mse_u['mse']+val_mse_v['mse'])/std_uv**2 ), 100. * (1.-(val_mse_u['mse']+val_mse_v['mse'])/var_test_uv) ))
        print('....... norm Lpqr thr = %.2e / %.2e'%(np.sqrt(val_l_pqr),val_l_pqr))       

        print(' ')
        print('....... Test dataset')
        print('....... MSE Test dataset (SSH) : OI = %.3e -- 4DVarNN = %.3e / %.2f %%'%(test_mseOI['mse'],test_mseRec['mse'],100. * (1.-test_mseRec['mse']/test_mseOI['mse'])))
        print('....... MSE Test dataset (gSSH): OI = %.3e -- 4DVarNN = %.3e / %.2f / %.2f %%'%(test_mseOI['mseGrad'],test_mseRec['mseGrad'],100. * (1.-test_mseRec['mseGrad']/test_mseOI['meanGrad']),100. * (1.-test_mseRec['mseGrad']/test_mseOI['mseGrad'])))
        print('....... (R)MSE Test dataset (u,v): %.2e / %.2e / %.2f %%/ %.2f %%'%(np.sqrt(test_mse_u['mse']+test_mse_v['mse']),test_mse_u['mse']+test_mse_v['mse'],100. * (1.-(test_mse_u['mse']+test_mse_v['mse'])/std_uv**2 ),100. * (1.-(test_mse_u['mse']+test_mse_v['mse'])/var_test_uv )))
        print('....... norm Lpqr thr = %.2e / %.2e / %.2f %%'%(np.sqrt(test_l_pqr),test_l_pqr,100.*(1.-test_l_pqr/test_var_l_pqr)))
        print('....... Mean current magnitude: %.2e / %.2e / %.2e'%(np.mean( np.sqrt(u_gt**2 + v_gt**2) ),np.mean( (u_gt**2 + v_gt**2) ),test_var_l_pqr))