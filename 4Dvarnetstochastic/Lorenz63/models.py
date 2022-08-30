import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats

import Updated_Solver as NN_4DVar 
from metrics import save_NetCDF, save_NetCDF2, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, animate_maps, plot_ensemble, L_hat_opti, K_hat, K_hat_tensor
#from new_dataloading import m_seuil, Indtest, debit_gt,debit_obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(torch.nn.Module):
    def __init__(self,dimIn, DimAE,dW):
        super(Encoder, self).__init__()
        self.pool1  = torch.nn.AvgPool1d(4)
        self.conv1  = NN_4DVar.ConstrainedConv1d(dimIn,2*dimIn*DimAE,2*dW+1,padding=dW,bias=False)
        self.conv2  = torch.nn.Conv1d(2*dimIn*DimAE,dimIn*DimAE,1,padding=0,bias=False)
          
        self.conv21 = torch.nn.Conv1d(dimIn*DimAE,dimIn*DimAE,1,padding=0,bias=False)
        self.conv22 = torch.nn.Conv1d(dimIn*DimAE,dimIn*DimAE,1,padding=0,bias=False)
        self.conv23 = torch.nn.Conv1d(dimIn*DimAE,dimIn*DimAE,1,padding=0,bias=False)
        self.conv3  = torch.nn.Conv1d(2*dimIn*DimAE,dimIn*DimAE,1,padding=0,bias=False)
        #self.conv4 = torch.nn.Conv1d(4*shapeData[0]*DimAE,8*shapeData[0]*DimAE,1,padding=0,bias=False)

        self.conv2Tr = torch.nn.ConvTranspose1d(dimIn*DimAE,dimIn,4,stride=4,bias=False)          
          #self.conv5 = torch.nn.Conv1d(2*shapeData[0]*DimAE,2*shapeData[0]*DimAE,3,padding=1,bias=False)
          #self.conv6 = torch.nn.Conv1d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)
          #self.conv6 = torch.nn.Conv1d(16*shapeData[0]*DimAE,shapeData[0],3,padding=1,bias=False)

        self.convHR1  = NN_4DVar.ConstrainedConv1d(dimIn,2*dimIn*DimAE,2*dW+1,padding=dW,bias=False)
        self.convHR1  = NN_4DVar.ConstrainedConv1d(dimIn,2*dimIn*DimAE,2*dW+1,padding=dW,bias=False)
        self.convHR2  = torch.nn.Conv1d(2*dimIn*DimAE,dimIn*DimAE,1,padding=0,bias=False)
          
        self.convHR21 = torch.nn.Conv1d(dimIn*DimAE,dimIn*DimAE,1,padding=0,bias=False)
        self.convHR22 = torch.nn.Conv1d(dimIn*DimAE,dimIn*DimAE,1,padding=0,bias=False)
        self.convHR23 = torch.nn.Conv1d(dimIn*DimAE,dimIn*DimAE,1,padding=0,bias=False)
        self.convHR3  = torch.nn.Conv1d(2*dimIn*DimAE,dimIn,1,padding=0,bias=False)

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
        
        return x





class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, x):
        return torch.mul(1., x)


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

class Phi_r(torch.nn.Module):
    def __init__(self, shapeData, DimAE, dW):
        super(Phi_r, self).__init__()
        self.encoder = Encoder(shapeData, DimAE, dW)
        self.decoder = Decoder()
        self.correlate_noise = CorrelateNoise(shapeData, 10)
        self.regularize_variance = RegularizeVariance(shapeData, 10)
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Gradient_img(torch.nn.Module):
    def __init__(self):
        super(Gradient_img, self).__init__()

        a = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])
        self.convGx = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGx.weight = torch.nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0),
                                                requires_grad=False)

        b = np.array([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])
        self.convGy = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.convGy.weight = torch.nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0),
                                                requires_grad=False)


        #self.eps=10**-6
        self.eps=0.

    def forward(self, im):

        if im.size(1) == 1:
            G_x = self.convGx(im)
            G_y = self.convGy(im)
            G = torch.sqrt(torch.pow(0.5 * G_x, 2) + torch.pow(0.5 * G_y, 2) + self.eps)
        else:

            for kk in range(0, im.size(1)):
                G_x = self.convGx(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))
                G_y = self.convGy(im[:, kk, :, :].view(-1, 1, im.size(2), im.size(3)))

                G_x = G_x.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                G_y = G_y.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                nG = torch.sqrt(torch.pow(0.5 * G_x, 2) + torch.pow(0.5 * G_y, 2)+ self.eps)

                if kk == 0:
                    G = nG.view(-1, 1, im.size(2) - 2, im.size(3) - 2)
                else:
                    G = torch.cat((G, nG.view(-1, 1, im.size(2) - 2, im.size(3) - 2)), dim=1)
        return G

class ModelLR(torch.nn.Module):
    def __init__(self):
        super(ModelLR, self).__init__()

        self.pool = torch.nn.AvgPool2d((16, 16))

    def forward(self, im):
        return self.pool(im)
  

class Model_H(torch.nn.Module):
    def __init__(self, shapeData):
        super(Model_H, self).__init__()
        self.DimObs = 1
        self.dimObsChannel = np.array([shapeData])

    def forward(self, x, y, mask):
        dyout = (x[:,:3,:] - y) * mask
        return dyout


      
class HParam:
    def __init__(self):
        self.iter_update     = []
        self.nb_grad_update  = []
        self.lr_update       = []
        self.n_grad          = 1
        self.dim_grad_solver = 10
        self.dropout         = 0.25
        self.automatic_optimization = True
        self.k_batch         = 1
        
        self.GradType       = 1 
        self.OptimType      = 2 

        self.alpha_proj    = 0.5
        self.alpha_sr      = 0.5
        self.alpha_lr      = 0.5  # 1e4
     
        self.flag_median_output = False
        #self.median_filter_width = width_med_filt_spatial
        self.dw_loss = 32

        
                          
        self.alpha          = np.array([1.,0.1])
        self.alpha4DVar     = np.array([0.01,1.])#np.array([0.30,1.60])#

        self.flagLearnWithObsOnly = False #True # 
        self.lambda_LRAE          = 0.5 # 0.5

        self.GradType       = 1 # Gradient computation (0: subgradient, 1: true gradient/autograd)
        self.OptimType      = 2 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
        
        self.NbProjection   = [0,0,0,0,0,0,0]#[0,0,0,0,0,0]#[5,5,5,5,5]##
        self.NBProjCurrent =0
        self.DT=13
       
    
    
class LitModel(pl.LightningModule):
    def __init__(self,conf=HParam(),*args, **kwargs):
        super().__init__()
        
        # hyperparameters
        self.hparams.alpha          = np.array([1.,1.])
        self.hparams.alpha4DVar     = np.array([0.01,1.])#np.array([0.30,1.60])#

        self.hparams.flagLearnWithObsOnly = False #True # 
        self.hparams.lambda_LRAE          = 0.5 # 0.5

        self.hparams.GradType       = 1 # Gradient computation (0: subgradient, 1: true gradient/autograd)
        self.hparams.OptimType      = 2 # 0: fixed-step gradient descent, 1: ConvNet_step gradient descent, 2: LSTM-based descent
        
        self.hparams.NbProjection   = [0,0,0,0,0,0,0]#[0,0,0,0,0,0]#[5,5,5,5,5]##
        self.hparams.iter_update     = [0,100,200,300,500,600]  # [0,2,4,6,9,15]
        self.hparams.nb_grad_update  = [10,10,10,15,20,20,20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.hparams.lr_update       = [1e-3,1e-3,1e-4,1e-5,1e-5,1e-5,1e-6,1e-6,1e-7]
        self.hparams.k_batch         = 1
        
         
        self.hparams.n_grad          = self.hparams.nb_grad_update[0]
        
        self.hparams.alpha_proj    = 0.5
        self.hparams.alpha_sr      = 0.5
        self.hparams.alpha_lr      = 0.5  # 1e4
        self.hparams.DT = 10
        
        self.hparams.automatic_optimization = False#True#        
        self.hparams.NBProjCurrent =self.hparams.NbProjection[0]
        
        #A revoir
        self.hparams.DimAE=20#150 avant
        self.hparams.dW = 3
        self.hparams.dW2=1
        self.hparams.sS = 2
        self.hparams.nbBlocks=1
        self.hparams.dropout_phi_r=0
        
        self.hparams.UsePriodicBoundary = False
        self.hparams.dim_grad_solver=20
        self.hparams.dropout =0.25
        self.hparams.stochastic = False
        
        #self.shapeData =[6,200] #for flagInit == 2
        self.shapeData = [9,200] #for flagInit == 3
        
        #self.std_Tr = kwargs['std_Tr']
        #self.mean_Tr = kwargs['mean_Tr']
        
        
        #lrCurrent       = lrUpdate[0] ??
        """    
        # main model
        self.model           = NN_4DVar.Model_4DVarNN_GradFP(phi_r,shapeData,self.hparams.NBProjCurrent,self.hparams.n_grad,self.hparams.GradType,self.hparams.OptimType,InterpFlag,UsePriodicBoundary)                
        """
        
        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.shapeData[0], self.hparams.DimAE, self.hparams.dW),
            Model_H(self.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(self.shapeData, self.hparams.UsePriodicBoundary,
                                          self.hparams.dim_grad_solver, self.hparams.dropout),
            None, None, self.shapeData, self.hparams.n_grad, self.hparams.stochastic)
        
        self.save_hyperparameters()
        self.model_LR = ModelLR()

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
            print("new lrCurrent")
            print(lrCurrent)
            lr = np.array([lrCurrent,lrCurrent,lrCurrent,0.5*lrCurrent,lrCurrent,0.])            
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]# * self.hparams.learning_rate
                mm += 1
                
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        opt = self.optimizers()
                    
        # compute loss and metrics    
        loss, out,input_init,t_gt, metrics,masks = self.compute_loss(train_batch, phase='train')

        # log step metric        
        
        self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
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
        loss, out, input_init,t_gt, metrics,masks = self.compute_loss(val_batch, phase='val')
        print(loss)
        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        opt = self.optimizers()
        print(opt.param_groups[1]['lr'])
        
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out,input_init,t_gt, metrics,masks = self.compute_loss(test_batch, phase='test')
        self.log('test_loss', loss)
        self.log("test_mse", metrics['mse'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        out_debit = out
        return {'preds': out[:,:3,:].detach().cpu(),
                'p_diag' : out[:,3:6,:].detach().cpu(),
                'p_outdiag' : out[:,6:,:].detach().cpu(),
                'init'      : input_init[:,:3,:].detach().cpu(),
                'targets'      : t_gt.detach().cpu(), 
                'masks'       : masks.detach().cpu()
               }

    def training_epoch_end(self, training_step_outputs):
        # do something with all training_step outputs
        print('.. \n')
    
    
    ##test epoch end prevision    
    def test_epoch_end(self, outputs):
        preds = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        p_diag = torch.cat([chunk['p_diag'] for chunk in outputs]).numpy()
        p_outdiag = torch.cat([chunk['p_outdiag'] for chunk in outputs]).numpy()
        inits = torch.cat([chunk['init'] for chunk in outputs]).numpy()
        targets = torch.cat([chunk['targets'] for chunk in outputs]).numpy()
        masks  = torch.cat([chunk['masks'] for chunk in outputs]).numpy()
        print(preds.shape)
        print("shapetest")
        
        path_save1 = self.logger.log_dir + '/test.nc'
        save_NetCDF(path_save1, 
                        preds = preds,
                        p_diag = p_diag,
                        p_outdiag = p_outdiag,
                        init = inits,
                        target = targets,
                        mask =masks
                    )
        
        return 1.

    def compute_loss(self, batch, phase):

        inputs_init,inputs_missing,masks,targets_GT = batch
        
       # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)
            

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, masks)
            
            
            #flagInit 3 :
            
            mean_state  =  outputs[:,:3,:]
            mean_state = torch.transpose(mean_state,1,2)
            mean_state = torch.reshape(mean_state,(outputs.shape[0]*outputs.shape[2], 3,1))
            h_state = outputs[:,3:,:]
            h_state = torch.transpose(h_state, 1,2)
            
            ind_diag = torch.Tensor(([[0,1,2],[0,1,2]])).long()
            ind_sup  = torch.Tensor(([[0,0,1],[1,2,2]])).long()
            ind_inf  = torch.Tensor(([[1,2,2],[0,0,1]])).long()                 
            
            
            p_state = torch.zeros((outputs.shape[0], outputs.shape[2],3,3)).to(device)            
            p_state[:,:,ind_diag[0],ind_diag[1]] = h_state[:,:,:3]*5
            p_state[:,:,ind_sup[0],ind_sup[1]] = h_state[:,:,3:]
            p_state[:,:,ind_inf[0],ind_inf[1]] = h_state[:,:,3:]
                   
               
            p_state = torch.reshape(p_state, (outputs.shape[0]*outputs.shape[2],3,3))
            p_state = torch.bmm(p_state,p_state)
            
            
            new_targets_GT =torch.zeros(targets_GT.shape[0], outputs.shape[1], targets_GT.shape[2]).to(device)
            new_targets_GT[:,:3,:] = targets_GT
            new_targets_GT[:,3:,:] = outputs[:,3:,:]  
            
            compared_targets_GT = torch.transpose(targets_GT,1,2)
            compared_targets_GT  = torch.reshape(compared_targets_GT,(outputs.shape[0]*outputs.shape[2], 3,1))
            dyn_out = self.model.phi_r(outputs)
            dyn_true = self.model.phi_r(new_targets_GT) 
                   
            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()
                mean_state = mean_state.detach()
                h_state  = h_state.detach()
                p_state = p_state.detach()
                new_targets_GT = new_targets_GT.detach()
                compared_targets_GT =compared_targets_GT.detach()
                masks =masks.detach()
                dyn_out= dyn_out.detach()
                dyn_true = dyn_true.detach()
                                      
            l1 = torch.bmm(p_state,mean_state-compared_targets_GT)
            l1 = torch.bmm(torch.transpose(mean_state-compared_targets_GT,1,2),l1)
            
            det = torch.linalg.det(p_state)
            loss_ML      = torch.mean(l1-torch.log(det+0.0001))
            '''
            #flagInit 2
            mean_state  =  outputs[:,:3,:]
            h_state = outputs[:,3:,:]
                        
            new_targets_GT =torch.zeros(targets_GT.shape[0], outputs.shape[1], targets_GT.shape[2]).to(device)
            new_targets_GT[:,:3,:] = targets_GT
            new_targets_GT[:,3:,:] = outputs[:,3:,:]  
            
            dyn_out = self.model.phi_r(outputs)
            dyn_true = self.model.phi_r(new_targets_GT) 

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()
                mean_state = mean_state.detach()
                h_state  = h_state.detach()
                new_targets_GT = new_targets_GT.detach()
                dyn_out =dyn_out.detach()
                dyn_true = dyn_true.detach()
            
            loss_c      = torch.mean((((mean_state - targets_GT)*torch.abs(h_state)*5)**2-2*torch.log(torch.abs(h_state)*5)))
                                                     
            loss_ML      = loss_c
            '''
            loss_R      = torch.sum((outputs[:,:3,:] - targets_GT)**2* masks )
            loss_R      = torch.mul(1.0 / torch.sum(masks),loss_R)
            
            loss_I      = torch.sum((outputs[:,:3,:] - targets_GT)**2* (1. - masks) )
            loss_I      = torch.mul(1.0 / torch.sum(1.-masks),loss_I)
            
            #loss_AE_cov =  torch.mean((self.model.phi_r(out) - outputs)**2 )
            loss_AE_mu     = torch.mean((dyn_out[:,:3,:] - outputs[:,:3,:])**2 )
            loss_AE_GT_mu  = torch.mean((dyn_true[:,:3,:] - new_targets_GT[:,:3,:])**2 )
            loss_AE_cov    = torch.mean((dyn_out[:,3:,:] - outputs[:,3:,:])**2 )
            loss_AE_GT_cov = torch.mean((dyn_true[:,3:,:] - new_targets_GT[:,3:,:])**2 )
            
            loss_squared = loss_I+loss_R
            
                               
            
            loss         = 0.01* self.hparams.alpha[0] * loss_ML +loss_squared+ 0.1 * self.hparams.alpha[1] * ( loss_AE_mu + loss_AE_GT_mu )+0.1*(loss_AE_cov+loss_AE_GT_cov)
            
                       
            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()
                inputs_init = inputs_init.detach()
                targets_GT = targets_GT.detach()
                #l1 = l1.detach()
                print("loss AE")
                print(loss_AE_mu)
                print("loss AE GT")
                print(loss_AE_GT_mu)
                print("loss ML")
                print(loss_ML)
                print("loss cov")
                print(loss_AE_cov+loss_AE_GT_cov)
                print("loss_squared")
                print(loss_I)
                print(loss_R)
                #print("loss rc")
                #print(torch.mean((outputs[:,3:,:]-inputs_init[:,3:,:])**2))
                #print("p-state")
                #print(p_state[0,:,:])
                #print("loss_ML")
                #print(torch.mean(l1-torch.log(det)))
                
                print("outputs")
                print(outputs[0,:,0])
              
            
            if np.isnan(loss.detach().cpu()) :
                print(loss_ML)
                print(p_state[torch.where(torch.isnan(torch.log(det)))])
                print(det[torch.where(det<=0)])
            
            # metrics
            mse = loss_ML.detach()
            metrics   = dict([('mse',mse)])
            #print(mse.cpu().detach().numpy())
            
                        
            outputs = outputs
        return loss,outputs,inputs_init,targets_GT, metrics,masks
        
    
#########################################################################################################################    
#End
#################################################################################################################
