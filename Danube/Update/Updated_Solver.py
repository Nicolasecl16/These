#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020

@author: rfablet
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

class ConvLSTM1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM1d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv1d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

def compute_WeightedLoss(x2,w):
    x2_msk = x2[:, w==1, ...]
    x2_num = ~x2_msk.isnan() & ~x2_msk.isinf()
    loss2 = F.mse_loss(x2_msk[x2_num], torch.zeros_like(x2_msk[x2_num]))
    loss2 = loss2 *  w.sum()
    return loss2


# Modules for the definition of the norms for
# the observation and prior model
class Model_WeightedL2Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL2Norm, self).__init__()
 
    def forward(self,x,w,eps=0.):
        loss_ = torch.nansum( x**2 , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_

    
class Model_GaussNorm(torch.nn.Module):
    def __init__(self):
        super(Model_GaussNorm, self).__init__()
#Dans ce cas, l'entrée x est un vecteur contenant l'état estimé et la covariance de l'erreur modèle que l'on suppose identique pour chaque station   
    def forward(self,x,w,cov,eps=0.):
        s     = x.shape
        loss_ = torch.nansum( x**2 , dim = 3)
        loss_ = torch.nansum( loss_*cov , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_reg = torch.log(torch.sum(cov))
        loss_ = torch.nansum( (loss_+loss_reg) * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_
    
class Model_GPDNorm(torch.nn.Module):
    def __init__(self):
        super(Model_GPDNorm, self).__init__()
#Dans ce cas, l'entrée x est un vecteur contenant l'état estimé et les paramètres de la GPD que l'on fait varier pour chaque pas de temps et pour chaque station

    def forward(self,x,ksi,sigma,w,eps=0.):
        s     = x.shape
        b = torch.zeros(s)+1
        borne_inf= 0.00001*b
        borne_inf=borne_inf.to(device)
        loss_reg = torch.nansum( torch.log(sigma))
        loss_i = torch.nansum((1+1/ksi)*torch.log(torch.maximum(1+ksi*(x)/sigma,borne_inf)))
        loss_  = loss_reg + loss_i
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_
    
    

class Model_WeightedL1Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()
 
    def forward(self,x,w,eps):

        loss_ = torch.nansum( torch.sqrt( eps**2 + x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_

class Model_WeightedLorenzNorm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedLorenzNorm, self).__init__()
 
    def forward(self,x,w,eps):

        loss_ = torch.nansum( torch.log( 1. + eps**2 * x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_

class Model_WeightedGMcLNorm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()
 
    def forward(self,x,w,eps):

        loss_ = torch.nansum( 1.0 - torch.exp( - eps**2 * x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_

def compute_WeightedL2Norm1D(x2,w):
    loss_ = torch.nansum(x2**2 , dim = 2)
    loss_ = torch.nansum( loss_ , dim = 0)
    loss_ = torch.nansum( loss_ * w )
    loss_ = loss_ / (torch.sum(~torch.isnan(x2)) / x2.shape[1] )
    
    return loss_

# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class model_GradUpdateLSTM(torch.nn.Module):
    def __init__(self,ShapeData,periodicBnd=False,DimLSTM=0,rateDropout=0.):
        super(model_GradUpdateLSTM, self).__init__()

        with torch.no_grad():
            self.shape     = ShapeData
            if DimLSTM == 0 :
                self.DimState  = 5*self.shape[0]
            else :
                self.DimState  = DimLSTM
            self.PeriodicBnd = periodicBnd
            if( (self.PeriodicBnd == True) & (len(self.shape) == 2) ):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False

        self.convLayer     = self._make_ConvGrad()
        K = torch.Tensor([0.1]).view(1,1,1,1)
        self.convLayer.weight = torch.nn.Parameter(K)

        self.dropout = torch.nn.Dropout(rateDropout)

        if len(self.shape) == 2: ## 1D Data
            self.lstm = ConvLSTM1d(self.shape[0],self.DimState,3)
        elif len(self.shape) == 3: ## 2D Data
            self.lstm = ConvLSTM2d(self.shape[0],self.DimState,3)

    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(self.DimState, self.shape[0], 1, padding=0,bias=False))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(torch.nn.Conv2d(self.DimState, self.shape[0], (1,1), padding=0,bias=False))

        return torch.nn.Sequential(*layers)
    def _make_LSTMGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(ConvLSTM1d(self.shape[0],self.DimState,3))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(ConvLSTM2d(self.shape[0],self.DimState,3))

        return torch.nn.Sequential(*layers)

    def forward(self,hidden,cell,grad,gradnorm=1.0):

        # compute gradient
        grad  = grad / gradnorm
        grad  = self.dropout( grad )

        if self.PeriodicBnd == True :
            dB     = 7
            #
            grad_  = torch.cat((grad[:,:,grad.size(2)-dB:,:],grad,grad[:,:,0:dB,:]),dim=2)
            if hidden is None:
                hidden_,cell_ = self.lstm(grad_,None)
            else:
                hidden_  = torch.cat((hidden[:,:,grad.size(2)-dB:,:],hidden,hidden[:,:,0:dB,:]),dim=2)
                cell_    = torch.cat((cell[:,:,grad.size(2)-dB:,:],cell,cell[:,:,0:dB,:]),dim=2)
                hidden_,cell_ = self.lstm(grad_,[hidden_,cell_])

            hidden = hidden_[:,:,dB:grad.size(2)+dB,:]
            cell   = cell_[:,:,dB:grad.size(2)+dB,:]
        else:
            if hidden is None:
                hidden,cell = self.lstm(grad,None)
            else:
                hidden,cell = self.lstm(grad,[hidden,cell])

        grad = self.dropout( hidden )
        grad = self.convLayer( grad )

        return grad,hidden,cell


# New module for the definition/computation of the variational cost
class Model_Var_Cost(nn.Module):
    def __init__(self ,m_NormObs, m_NormPhi, ShapeData,DimObs=1,dimObsChannel=0,dimState=0):
        super(Model_Var_Cost, self).__init__()
        self.dimObsChannel = dimObsChannel
        self.DimObs        = DimObs
        if dimState > 0 :
            self.DimState      = dimState
        else:
            self.DimState      = ShapeData[0]
            
        # parameters for variational cost
        self.alphaObs    = torch.nn.Parameter(torch.Tensor(1. * np.ones((self.DimObs,1))))
        self.alphaReg    = torch.nn.Parameter(torch.Tensor([1.]))
        if self.dimObsChannel[0] == 0 :
            self.WObs           = torch.nn.Parameter(torch.Tensor(np.ones((self.DimObs,ShapeData[0]))))
            self.dimObsChannel  = ShapeData[0] * np.ones((self.DimObs,))
        else:
            self.WObs            = torch.nn.Parameter(torch.Tensor(np.ones((self.DimObs,np.max(self.dimObsChannel)))))
        self.WReg    = torch.nn.Parameter(torch.Tensor(np.ones(self.DimState,)))
        self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.DimObs,))))
        self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))
        
        self.normObs   = m_NormObs
        self.normPrior = m_NormPhi
        
    def forward(self, dx, dy):

        loss = self.alphaReg**2 * self.normPrior(dx,self.WReg**2,self.epsReg)
                
        if self.DimObs == 1 :
            loss +=  self.alphaObs[0]**2 * self.normObs(dy,self.WObs[0,:]**2,self.epsObs[0])
        else:
            for kk in range(0,self.DimObs):
                loss +=  self.alphaObs[kk]**2 * self.normObs(dy[kk],self.WObs[kk,0:dy[kk].size(1)]**2,self.epsObs[kk])

        return loss

# New module for the definition/computation of the variational cost
class Model_Var_Cost_inside(nn.Module):
    def __init__(self ,m_NormObs, m_NormPhi, ShapeData,DimObs=1,dimObsChannel=0,dimState=0):
        super(Model_Var_Cost_inside, self).__init__()
        self.dimObsChannel = dimObsChannel
        self.DimObs        = DimObs
        if dimState > 0 :
            self.DimState      = dimState
        else:
            self.DimState      = ShapeData[0]
            
        # parameters for variational cost
        self.alphaObs    = torch.nn.Parameter(torch.Tensor(1. * np.ones((self.DimObs,1))))
        self.alphaReg    = torch.nn.Parameter(torch.Tensor([1.]))
        if self.dimObsChannel[0] == 0 :
            self.WObs           = torch.nn.Parameter(torch.Tensor(np.ones((self.DimObs,ShapeData[0]))))
            self.dimObsChannel  = ShapeData[0] * np.ones((self.DimObs,))
        else:
            self.WObs            = torch.nn.Parameter(torch.Tensor(np.ones((self.DimObs,np.max(self.dimObsChannel)))))
        self.WReg    = torch.nn.Parameter(torch.Tensor(np.ones(self.DimState,)))
        self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.DimObs,))))
        self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))
        
        self.normObs   = m_NormObs
        self.normPrior = m_NormPhi
        
    def forward(self, dx, dy,ksi,sigma):

        loss = self.alphaReg**2 * self.normPrior(dx,ksi,sigma,self.WReg**2,self.epsReg)
                
        if self.DimObs == 1 :
            loss +=  self.alphaObs[0]**2 * self.normObs(dy,self.WObs[0,:]**2,self.epsObs[0])
        else:
            for kk in range(0,self.DimObs):
                loss +=  self.alphaObs[kk]**2 * self.normObs(dy[kk],self.WObs[kk,0:dy[kk].size(1)]**2,self.epsObs[kk])

        return loss    
    
class CorrelateNoise(torch.nn.Module):
    def __init__(self, shape_data, dim_cn):
        super(CorrelateNoise, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data, dim_cn, (3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(dim_cn, 2 * dim_cn, (3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim_cn, shape_data, (3, 3), padding=1, bias=False)

    def forward(self, w):
        w = self.conv1(F.relu(w)).to(device)
        w = self.conv2(F.relu(w)).to(device)
        w = self.conv3(w).to(device)
        return w


class RegularizeVariance(torch.nn.Module):
    def __init__(self, shape_data, dim_rv):
        super(RegularizeVariance, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data, dim_rv, (3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(dim_rv, 2 * dim_rv, (3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim_rv, shape_data, (3, 3), padding=1, bias=False)

    def forward(self, v):
        v = self.conv1(F.relu(v)).to(device)
        v = self.conv2(F.relu(v)).to(device)
        v = self.conv3(v).to(device)
        return v


# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters 
# (default norm (None) refers to the L2 norm)
# updated inner modles to account for the variational model module
class Solver_Grad_4DVarNN(nn.Module):
    def __init__(self ,phi_r,mod_H, m_Grad, m_NormObs, m_NormPhi, ShapeData,n_iter_grad, stochastic=False):
        super(Solver_Grad_4DVarNN, self).__init__()
        self.phi_r         = phi_r
        
        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        if m_NormPhi == None:    
            m_NormPhi = Model_WeightedL2Norm()
            
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost(m_NormObs, m_NormPhi, ShapeData,mod_H.DimObs,mod_H.dimObsChannel)

        self.correlate_noise = CorrelateNoise(ShapeData[0], 10)
        self.regularize_variance = RegularizeVariance(ShapeData[0], 10)
        self.stochastic = stochastic

        with torch.no_grad():
            self.n_grad = int(n_iter_grad)
        
    def forward(self, x, yobs, mask):
        return self.solve(
            x_0=x,
            obs=yobs,
            mask = mask)

    def solve(self, x_0, obs, mask):
        x_k = torch.mul(x_0,1.) 
        hidden = None
        cell = None 
        normgrad_ = 0.
        
        for _ in range(self.n_grad):
            x_k_plus_1, hidden, cell, normgrad_ = self.solver_step(x_k, obs, mask,hidden, cell, normgrad_)

            x_k = torch.mul(x_k_plus_1,1.)

        return x_k_plus_1, hidden, cell, normgrad_

    def solver_step(self, x_k, obs, mask, hidden, cell,normgrad = 0.):
        var_cost, var_cost_grad= self.var_cost(x_k, obs, mask)
        if normgrad == 0. :
            normgrad_= torch.sqrt( torch.mean( var_cost_grad**2 + 0.))
        else:
            normgrad_= normgrad
        grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        grad *= 1./ self.n_grad
        '''
        if self.stochastic == True:
            W = torch.randn(x_k.shape).to(device)
            gW = torch.mul(self.regularize_variance(x_k),self.correlate_noise(W))
            grad = grad + gW
        '''
        x_k_plus_1 = x_k - grad
        return x_k_plus_1, hidden, cell, normgrad_

    def var_cost(self , x, yobs, mask):
        dy = self.model_H(x,yobs,mask)
        dx = x - self.phi_r(x)
        
        loss = self.model_VarCost( dx , dy )
        
        var_cost_grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        return loss, var_cost_grad
    
    
    
class Solver_Grad_4DVarNN_parametric(nn.Module):
    def __init__(self ,phi_r,mod_H, Cov_model, m_Grad, m_NormObs, m_NormPhi, ShapeData,n_iter_grad, stochastic=False):
        super(Solver_Grad_4DVarNN_parametric, self).__init__()
        self.phi_r         = phi_r
        
        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        if m_NormPhi == None:    
            m_NormPhi = Model_WeightedL2Norm()
        
        
        self.model_H = mod_H
        self.Covariance_model = Cov_model
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost(m_NormObs, m_NormPhi, ShapeData,mod_H.DimObs,mod_H.dimObsChannel)

        self.correlate_noise = CorrelateNoise(ShapeData[0], 10)
        self.regularize_variance = RegularizeVariance(ShapeData[0], 10)
        self.stochastic = stochastic

        with torch.no_grad():
            self.n_grad = int(n_iter_grad)
        
    def forward(self, x, yobs, mask):
        
        return self.solve(
            x_0=x,
            obs=yobs,
            mask = mask)

    def solve(self, x_0, obs, mask):
        x_k = torch.mul(x_0,1.) 
        hidden = None
        cell = None 
        normgrad_ = 0.
        
        for _ in range(self.n_grad):
            x_k_plus_1, hidden, cell, normgrad_ = self.solver_step(x_k, obs, mask,hidden, cell, normgrad_)

            x_k = torch.mul(x_k_plus_1,1.)

        return x_k_plus_1, hidden, cell, normgrad_

    def solver_step(self, x_k, obs, mask, hidden, cell,normgrad = 0.):
        var_cost, var_cost_grad= self.var_cost(x_k, obs, mask)
        if normgrad == 0. :
            normgrad_= torch.sqrt( torch.mean( var_cost_grad**2 + 0.))
        else:
            normgrad_= normgrad
        grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        grad *= 1./ self.n_grad
        '''
        if self.stochastic == True:
            W = torch.randn(x_k.shape).to(device)
            gW = torch.mul(self.regularize_variance(x_k),self.correlate_noise(W))
            grad = grad + gW
        '''
        x_k_plus_1 = x_k - grad
        return x_k_plus_1, hidden, cell, normgrad_

    def var_cost(self , x, yobs, mask):
        dy = self.model_H(x,yobs,mask)
        dx = x - self.phi_r(x)
        
        loss = self.model_VarCost( dx , dy )
        
        var_cost_grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        return loss, var_cost_grad


#On abandonne l'idée d'une    
class Solver_Grad_4DVarNN_inside(nn.Module):
    def __init__(self ,phi_r,mod_H, m_Grad, m_NormObs, m_NormPhi, ShapeData,n_iter_grad, stochastic=False):
        super(Solver_Grad_4DVarNN_inside, self).__init__()
        self.phi_r         = phi_r
        
        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        if m_NormPhi == None:    
            m_NormPhi = Model_GPDNorm()
            
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost_inside(m_NormObs, m_NormPhi, ShapeData,mod_H.DimObs,mod_H.dimObsChannel)

        self.correlate_noise = CorrelateNoise(ShapeData[0], 10)
        self.regularize_variance = RegularizeVariance(ShapeData[0], 10)
        self.stochastic = stochastic

        with torch.no_grad():
            self.n_grad = int(n_iter_grad)
        
    def forward(self, x, yobs, mask):
        return self.solve(
            x_0=x,
            obs=yobs,
            mask = mask)

    def solve(self, x_0, obs, mask):
        x_k = torch.mul(x_0,1.) 
        hidden = None
        cell = None 
        normgrad_ = 0.
        
        for _ in range(self.n_grad):
            x_k_plus_1, hidden, cell, normgrad_ = self.solver_step(x_k, obs, mask,hidden, cell, normgrad_)

            x_k = torch.mul(x_k_plus_1,1.)

        return x_k_plus_1, hidden, cell, normgrad_

    def solver_step(self, x_k, obs, mask, hidden, cell,normgrad = 0.):
        var_cost, var_cost_grad= self.var_cost(x_k, obs, mask)
        if normgrad == 0. :
            normgrad_= torch.sqrt( torch.mean( var_cost_grad**2 + 0.))
        else:
            normgrad_= normgrad
        grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        grad *= 1./ self.n_grad
        '''
        if self.stochastic == True:
            W = torch.randn(x_k.shape).to(device)
            gW = torch.mul(self.regularize_variance(x_k),self.correlate_noise(W))
            grad = grad + gW
        '''
        x_k_plus_1 = x_k - grad
        return x_k_plus_1, hidden, cell, normgrad_

    def var_cost(self , x, yobs, mask):
        phi_x=self.phi_r(x)
        mu  = phi_x[:,:,:,:12]        
        ksi = phi_x[:,:,:,12]+0.0001
        ksi = torch.unsqueeze(ksi,dim=3)
        ksi = ksi.expand(-1,-1,-1,12)
        sigma = phi_x[:,:,:,13]+0.0001
        sigma = torch.unsqueeze(sigma,dim=3)
        sigma = sigma.expand(-1,-1,-1,12)
        mu=mu.to(device)
        ksi = ksi.to(device)
        sigma = sigma.to(device)
        dy = self.model_H(x,yobs,mask)
        dx = x - mu
        loss = self.model_VarCost( dx , dy,ksi,sigma )
        
        var_cost_grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        return loss, var_cost_grad
    
    