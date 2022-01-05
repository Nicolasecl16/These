import einops
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from scipy import stats

import Updated_Solver as NN_4DVar 
from metrics_Copy1 import save_NetCDF, save_NetCDFparam, save_NetCDFparamGPD, nrmse_scores, mse_scores, plot_nrmse, plot_mse, plot_snr, plot_maps, animate_maps, plot_ensemble, L_hat_opti, K_hat, K_hat_tensor
from new_dataloading_Copy1 import m_seuil, Indtest, debit_gt,debit_obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class CovarianceNetwork(torch.nn.Module):
    def __init__(self, shapeData, dimout, dimH):
        super(CovarianceNetwork,self).__init__()
        self.layer1 = torch.nn.Linear(shapeData, dimH)
        self.layer2 = torch.nn.Linear(dimH, dimout)

    def forward(self, w):
        shape = w.shape       
        w=torch.reshape(w,(shape[0],shape[1],shape[2]*shape[3])).to(device)
        w = self.layer1(F.relu(w)).to(device)
        w = self.layer2(F.relu(w)).to(device)
        w = (torch.abs(w)).to(device)
        w=torch.reshape(w,(shape[0],shape[1],shape[2],shape[3])).to(device)
        return w

class BiLinUnit(torch.nn.Module):
    def __init__(self, dimIn, dim, dW, dW2, dropout=0.):
        super(BiLinUnit, self).__init__()
        self.conv1 = torch.nn.Conv2d(dimIn, 2 * dim, (2 * dW + 1, 2 * dW + 1), padding=dW, bias=False)
        self.conv2 = torch.nn.Conv2d(2 * dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim, dimIn, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.bilin0 = torch.nn.Conv2d(dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.bilin1 = torch.nn.Conv2d(dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.bilin2 = torch.nn.Conv2d(dim, dim, (2 * dW2 + 1, 2 * dW2 + 1), padding=dW2, bias=False)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, xin):
        x = self.conv1(xin)
        print(x.shape)
        x = self.dropout(x)
        x = self.conv2(F.relu(x))
        print(x.shape)
        x = self.dropout(x)
        x = torch.cat((self.bilin0(x), self.bilin1(x) * self.bilin2(x)), dim=1)
        print(x.shape)
        x = self.dropout(x)
        x = self.conv3(x)
        print(x.shape)
        return x


class Encoder(torch.nn.Module):
    def __init__(self, dimInp, dimAE, dW, dW2, sS, nbBlocks, rateDropout=0.):
        super(Encoder, self).__init__()

        self.NbBlocks = nbBlocks
        self.DimAE = dimAE
        # self.conv1HR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
        # self.conv1LR  = torch.nn.Conv2d(dimInp,self.DimAE,(2*dW+1,2*dW+1),padding=dW,bias=False)
        #self.pool1 = torch.nn.AvgPool2d(sS)
        self.pool1 = torch.nn.AvgPool2d((1,sS))
        #self.convTr = torch.nn.ConvTranspose2d(dimInp, dimInp, (sS, sS), stride=(sS, sS), bias=False)
        self.convTr = torch.nn.ConvTranspose2d(dimInp, dimInp, (1, sS), stride=(1, sS), bias=False)
        
        # self.NNtLR    = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)
        # self.NNHR     = self.__make_ResNet(self.DimAE,self.NbBlocks,rateDropout)
        self.NNLR = self.__make_BilinNN(dimInp, self.DimAE, dW, dW2, self.NbBlocks, rateDropout)
        self.NNHR = self.__make_BilinNN(dimInp, self.DimAE, dW, dW2, self.NbBlocks, rateDropout)
        self.dropout = torch.nn.Dropout(rateDropout)

    def __make_BilinNN(self, dimInp, dimAE, dW, dW2, Nb_Blocks=2, dropout=0.):
        layers = []
        layers.append(BiLinUnit(dimInp, dimAE, dW, dW2, dropout))
        for kk in range(0, Nb_Blocks - 1):
            layers.append(BiLinUnit(dimAE, dimAE, dW, dW2, dropout))
        return torch.nn.Sequential(*layers)

    def forward(self, xinp):
        ## LR component
        print(xinp.shape)
        print(self.pool1(xinp).shape)
        xLR = self.NNLR(self.pool1(xinp))
        xLR = self.dropout(xLR)
        xLR = self.convTr(xLR)
        # HR component
        xHR = self.NNHR(xinp)
        return xLR + xHR


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
    def __init__(self, shapeData, DimAE, dW, dW2, sS, nbBlocks, rateDr, stochastic=False):
        super(Phi_r, self).__init__()
        self.encoder = Encoder(shapeData, DimAE, dW, dW2, sS, nbBlocks, rateDr)
        self.decoder = Decoder()
        self.correlate_noise = CorrelateNoise(shapeData, 10)
        self.regularize_variance = RegularizeVariance(shapeData, 10)
        self.stochastic = stochastic

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        if self.stochastic == True:
            W = torch.randn(x.shape).to(device)
            #  g(W) = alpha(x)*h(W)
            gW = torch.mul(self.regularize_variance(x),self.correlate_noise(W))
            # variance
            gW = gW/torch.std(gW)
            # print(stats.describe(gW.detach().cpu().numpy()))
            x = x + gW
        return x
    
class Phi_r_inside(torch.nn.Module):
    def __init__(self, shapeData, DimAE, dW, dW2, sS, nbBlocks, rateDr, borneinf, stochastic=False):
        super(Phi_r_inside, self).__init__()
        self.encoder = Encoder(shapeData, DimAE, dW, dW2, sS, nbBlocks, rateDr)
        self.decoder = Decoder()
        self.correlate_noise = CorrelateNoise(shapeData, 10)
        self.regularize_variance = RegularizeVariance(shapeData, 10)
        self.stochastic = stochastic
        self.Cov = CovarianceNetwork(62,62,62)
        self.borneinf = borneinf

    def forward(self, x):
        b_inf = torch.unsqueeze(self.borneinf,dim=0)
        b_inf = torch.unsqueeze(b_inf,dim=1)
        b_inf = b_inf.expand(x.shape[0],-1,-1,12)
        b_inf = b_inf.to(device)
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.abs(x) + b_inf
        c = self.Cov(torch.zeros(x.shape[0],x.shape[1],x.shape[2],2)+1)
        x=torch.cat((x,c),dim = 3)
        if self.stochastic == True:
            W = torch.randn(x.shape).to(device)
            #  g(W) = alpha(x)*h(W)
            gW = torch.mul(self.regularize_variance(x),self.correlate_noise(W))
            # variance
            gW = gW/torch.std(gW)
            # print(stats.describe(gW.detach().cpu().numpy()))
            x = x + gW
        return x    

class Model_H(torch.nn.Module):
    def __init__(self, shapeData):
        super(Model_H, self).__init__()
        self.DimObs = 1
        self.dimObsChannel = np.array([shapeData])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout


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

"""
############################################ Lightning Module #######################################################################
class LitModel(pl.LightningModule):
    def __init__(self, hparam, *args, **kwargs):
        super().__init__()
        #print(isinstance(hparams, dict))
        #hparams = hparam if isinstance(hparam, dict) else OmegaConf.to_container(hparam, resolve=True)
        self.hparam = hparam
        #hparams = hparam
        #self.save_hyperparameters(hparams)
        #self.hparams=hparam
        print(type(self.hparam))
        print(self.hparam)
        print(hparam.dT)
        
        #self.save_hyperparameters(hparams)???
        self.var_Val = kwargs['var_Val']
        self.var_Tr = kwargs['var_Tr']
        self.var_Tt = kwargs['var_Tt']

        
        self.var_Val = kwargs['var_Val']
        self.var_Tr = kwargs['var_Tr']
        self.var_Tt = kwargs['var_Tt']
        self.mean_Val = kwargs['mean_Val']
        self.mean_Tr = kwargs['mean_Tr']
        self.mean_Tt = kwargs['mean_Tt']
        self.shapeData = [self.hparam.dT*2,self.hparam.NbStations ]

        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.shapeData[0], self.hparam.DimAE, self.hparam.dW, self.hparam.dW2, self.hparam.sS,
                  self.hparam.nbBlocks,self.hparam.dropout_phi_r, self.hparam.stochastic),
            Model_H(self.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(self.shapeData, self.hparam.UsePriodicBoundary,
                                          self.hparam.dim_grad_solver, self.hparam.dropout),
            None, None, self.shapeData, self.hparam.n_grad, self.hparam.stochastic)

        self.model_LR = ModelLR()
        self.gradient_img = Gradient_img()
        # loss weghing wrt time

        self.w_loss = torch.nn.Parameter(kwargs['w_loss'], requires_grad=False)  # duplicate for automatic upload to gpu
        self.x_gt = None  # variable to store Ground Truth
        self.x_oi = None  # variable to store OI
        self.x_rec = None  # variable to store output of test method
        self.test_figs = {}

        self.automatic_optimization = self.hparam.automatic_optimization

    def forward(self):
        return 1

    def configure_optimizers(self):

        optimizer = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparam.lr_update[0]},
                                {'params': self.model.model_VarCost.parameters(), 'lr': self.hparam.lr_update[0]},
                                {'params': self.model.phi_r.parameters(), 'lr': 0.5 * self.hparam.lr_update[0]},
                                ], lr=0.)

        return optimizer

    def on_epoch_start(self):
        # enfore acnd check some hyperparameters
        self.model.n_grad = self.hparam.n_grad

    def on_train_epoch_start(self):
        opt = self.optimizers()
        if (self.current_epoch in self.hparam.iter_update) & (self.current_epoch > 0):
            indx = self.hparam.iter_update.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f' % (
                self.current_epoch, self.hparam.nb_grad_update[indx], self.hparam.lr_update[indx]))

            self.hparam.n_grad = self.hparam.nb_grad_update[indx]
            self.model.n_grad = self.hparam.n_grad

            mm = 0
            lrCurrent = self.hparam.lr_update[indx]
            lr = np.array([lrCurrent, lrCurrent, 0.5 * lrCurrent, 0.])
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]  # * self.hparams.learning_rate
                mm += 1

    def training_step(self, train_batch, batch_idx, optimizer_idx=0):

        # compute loss and metrics    
        loss, out, metrics = self.compute_loss(train_batch, phase='train')
        if loss is None:
            return loss
        # log step metric        
        # self.log('train_mse', mse)
        # self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] / self.var_Tr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)

        # initial grad value
        if self.hparam.automatic_optimization == False:
            opt = self.optimizers()
            # backward
            self.manual_backward(loss)

            if (batch_idx + 1) % self.hparam.k_batch == 0:
                # optimisation step
                opt.step()

                # grad initialization to zero
                opt.zero_grad()

        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')
        if loss is None:
            return loss
        self.log('val_loss', loss)
        self.log("val_mse", metrics['mse'] / self.var_Val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)
        return loss.detach()

    def test_step(self, test_batch, batch_idx):

        targets_OI, inputs_Mask, inputs_obs, targets_GT = test_batch
        loss, out, metrics = self.compute_loss(test_batch, phase='test')
        if loss is not None:
            self.log('test_loss', loss)
            self.log("test_mse", metrics['mse'] / self.var_Tt, on_step=False, on_epoch=True, prog_bar=True)
            self.log("test_mseG", metrics['mseGrad'] / metrics['meanGrad'], on_step=False, on_epoch=True, prog_bar=True)

        return {'gt'    : (targets_GT.detach().cpu()*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'oi'    : (targets_OI.detach().cpu()*np.sqrt(self.var_Tr)) + self.mean_Tr,
                'preds' : (out.detach().cpu()*np.sqrt(self.var_Tr)) + self.mean_Tr}


    def test_epoch_end(self, outputs):

        gt = torch.cat([chunk['gt'] for chunk in outputs]).numpy()
        oi = torch.cat([chunk['oi'] for chunk in outputs]).numpy()
        pred = torch.cat([chunk['preds'] for chunk in outputs]).numpy()

        ds_size = {'time': self.ds_size_time,
                   'lon': self.ds_size_lon,
                   'lat': self.ds_size_lat,
                   }

        gt, oi, pred = map(
            lambda t: einops.rearrange(
                t,
                '(t_idx lat_idx lon_idx) win_time win_lat win_lon -> t_idx win_time (lat_idx win_lat) (lon_idx win_lon)',
                t_idx=ds_size['time'],
                lat_idx=ds_size['lat'],
                lon_idx=ds_size['lon'],
            ),
            [gt, oi, pred])

        self.x_gt = gt[:, int(self.hparam.dT / 2), :, :]
        self.x_oi = oi[:, int(self.hparam.dT / 2), :, :]
        self.x_rec = pred[:, int(self.hparam.dT / 2), :, :]

        # display map
        path_save0 = self.logger.log_dir + '/maps.png'
        fig_maps = plot_maps(
                self.x_gt[0],
                  self.x_oi[0],
                  self.x_rec[0],
                  self.lon, self.lat, path_save0)
        self.test_figs['maps'] = fig_maps
        # animate maps
        if self.hparam.animate == True:
            path_save0 = self.logger.log_dir + '/animation.mp4'
            animate_maps(self.x_gt,
                         self.x_oi,
                         self.x_rec,
                         self.lon, self.lat, path_save0)
        # compute nRMSE
        path_save2 = self.logger.log_dir + '/nRMSE.txt'
        tab_scores = nrmse_scores(gt, oi, pred, path_save2)
        print('*** Display nRMSE scores ***')
        print(tab_scores)

        path_save21 = self.logger.log_dir + '/MSE.txt'
        tab_scores = mse_scores(gt, oi, pred, path_save21)
        print('*** Display MSE scores ***')
        print(tab_scores)

        # plot nRMSE
        path_save3 = self.logger.log_dir + '/nRMSE.png'
        nrmse_fig = plot_nrmse(self.x_gt,  self.x_oi, self.x_rec, path_save3, index_test=np.arange(60, 60+self.ds_size_time))
        self.test_figs['nrmse'] = nrmse_fig

        # plot MSE
        path_save31 = self.logger.log_dir + '/MSE.png'
        mse_fig = plot_mse(self.x_gt, self.x_oi, self.x_rec, path_save31,
                               index_test=np.arange(60, 60 + self.ds_size_time))
        self.test_figs['mse'] = mse_fig
        self.logger.experiment.add_figure('Maps', fig_maps, global_step=self.current_epoch)
        self.logger.experiment.add_figure('NRMSE', nrmse_fig, global_step=self.current_epoch)
        self.logger.experiment.add_figure('MSE', mse_fig, global_step=self.current_epoch)

        # plot SNR
        path_save4 = self.logger.log_dir + '/SNR.png'
        snr_fig = plot_snr(self.x_gt, self.x_oi, self.x_rec, path_save4)
        self.test_figs['snr'] = snr_fig

        self.logger.experiment.add_figure('SNR', snr_fig, global_step=self.current_epoch)
        # save NetCDF
        path_save1 = self.logger.log_dir + '/test.nc'
        save_netcdf(saved_path1=path_save1, pred=pred,
                    lon=self.lon, lat=self.lat, index_test=np.arange(60, 77))

    def compute_loss(self, batch, phase):

        targets_OI, inputs_Mask, inputs_obs, targets_GT = batch
        # handle patch with no observation
        if inputs_Mask.sum().item() == 0:
            return (
                None,
                torch.zeros_like(targets_GT),
                dict([('mse', 0.), ('mseGrad', 0.), ('meanGrad', 1.), ('mseOI', 0.),
                      ('mseGOI', 0.)])
            )
        new_masks = torch.cat((1. + 0. * inputs_Mask, inputs_Mask), dim=1)
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), torch.zeros_like(targets_GT))
        inputs_init = torch.cat((targets_OI, inputs_obs), dim=1)
        inputs_missing = torch.cat((targets_OI, inputs_obs), dim=1)

        # gradient norm field
        g_targets_GT = self.gradient_img(targets_GT)
        # need to evaluate grad/backward during the evaluation and training phase for phi_r
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, new_masks)

            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()

            outputsSLRHR = outputs
            outputsSLR = outputs[:, 0:self.hparam.dT, :, :]
            outputs = outputsSLR + outputs[:, self.hparam.dT:, :, :]

            # reconstruction losses
            g_outputs = self.gradient_img(outputs)
            loss_All = NN_4DVar.compute_WeightedLoss((outputs - targets_GT), self.w_loss)

            loss_GAll = NN_4DVar.compute_WeightedLoss(g_outputs - g_targets_GT, self.w_loss)
            loss_OI = NN_4DVar.compute_WeightedLoss(targets_GT - targets_OI, self.w_loss)
            loss_GOI = NN_4DVar.compute_WeightedLoss(self.gradient_img(targets_OI) - g_targets_GT, self.w_loss)

            # projection losses
            loss_AE = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR) ** 2)
            yGT = torch.cat((targets_GT_wo_nan, outputsSLR - targets_GT_wo_nan), dim=1)
            # yGT        = torch.cat((targets_OI,targets_GT-targets_OI),dim=1)
            loss_AE_GT = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)

            # low-resolution loss
            loss_SR = NN_4DVar.compute_WeightedLoss(outputsSLR - targets_OI, self.w_loss)
            targets_GTLR = self.model_LR(targets_OI)
            loss_LR = NN_4DVar.compute_WeightedLoss(self.model_LR(outputs) - targets_GTLR, self.w_loss)

            # total loss
            loss = self.hparam.alpha_mse_ssh * loss_All + self.hparam.alpha_mse_gssh * loss_GAll
            loss += 0.5 * self.hparam.alpha_proj * (loss_AE + loss_AE_GT)
            loss += self.hparam.alpha_lr * loss_LR + self.hparam.alpha_sr * loss_SR

            # metrics
            mean_GAll = NN_4DVar.compute_WeightedLoss(g_targets_GT, self.w_loss)
            mse = loss_All.detach()
            mseGrad = loss_GAll.detach()
            metrics = dict([('mse', mse), ('mseGrad', mseGrad), ('meanGrad', mean_GAll), ('mseOI', loss_OI.detach()),
                            ('mseGOI', loss_GOI.detach())])

        return loss, outputs, metrics
        
        
"""
      
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
        #self.median_filter_width = width_med_filt_spatial
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
        self.DT=13
        self.w_loss          = []
        self.w_ = np.zeros(self.DT)
        self.w_[int(self.DT / 2)] = 1.
        self.wLoss = torch.Tensor(self.w_)
    
    
    
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
        self.hparams.alpha_mse_vv = 1
        self.hparams.DT = 13
        self.hparams.w_=np.zeros(self.hparams.DT)
        self.hparams.w_[int(self.hparams.DT / 2)] = 1.
      
        self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(self.hparams.w_), requires_grad=False)
        self.hparams.automatic_optimization = False#True#

        
        self.hparams.NBProjCurrent =self.hparams.NbProjection[0]
        #A revoir
        self.hparams.DimAE=20
        self.hparams.dW = 3
        self.hparams.dW2=1
        self.hparams.sS = 2
        self.hparams.nbBlocks=1
        self.hparams.dropout_phi_r=0
        self.hparams.stochastic=False
        self.hparams.UsePriodicBoundary = False
        self.hparams.dim_grad_solver=150
        self.hparams.dropout =0.25
        
        self.shapeData =[1,31,12]
        self.var_Val = kwargs['var_Val']
        self.var_Tr = kwargs['var_Tr']
        self.var_Tt = kwargs['var_Tt']
        self.meanTrtrue = kwargs['meanTrtrue']
        self.stdTrtrue = kwargs['stdTrtrue']
        
        #lrCurrent       = lrUpdate[0] ??
        """    
        # main model
        self.model           = NN_4DVar.Model_4DVarNN_GradFP(phi_r,shapeData,self.hparams.NBProjCurrent,self.hparams.n_grad,self.hparams.GradType,self.hparams.OptimType,InterpFlag,UsePriodicBoundary)                
        """
        
        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(self.shapeData[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS,
                  self.hparams.nbBlocks,self.hparams.dropout_phi_r, self.hparams.stochastic),
            Model_H(self.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(self.shapeData, self.hparams.UsePriodicBoundary,
                                          self.hparams.dim_grad_solver, self.hparams.dropout),
            None, None, self.shapeData, self.hparams.n_grad, self.hparams.stochastic)
        
        self.save_hyperparameters()
        self.model_LR = ModelLR()

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
        self.log("tr_mse", metrics['mse'] / self.var_Tr , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        self.log("val_mse", metrics['mse'] / self.var_Val , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_uv", metrics['mse_uv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics = self.compute_loss(test_batch, phase='test')

        self.log('test_loss', loss)
        self.log("test_mse", metrics['mse'] / self.var_Tt , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        x_test_rec = self.stdTrtrue * x_test_rec + self.meanTrtrue
        self.x_rec_debit = x_test_rec[:,0,:,:]
        print("shapetest")
        print(self.x_rec_debit.shape)
        print(debit_gt.shape)
        print(debit_obs.shape)
        path_save1 = self.logger.log_dir + '/test.nc'
        save_NetCDF(path_save1, 
                        Indtest,
                        debit_gt = debit_gt , 
                        debit_obs = debit_obs,
                        debit_rec = self.x_rec_debit,
                        mode='prevision'
                    )
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

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, masks)
            

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
                
            #X = (torch.moveaxis(outputs[:,0,:,int(2*self.hparams.DT/3+1):],0,1)).reshape(outputs.shape[2],outputs.shape[0]*(outputs.shape[3]-int(2*self.hparams.DT/3+1)))
            
            #Y=(torch.moveaxis(targets_GT[:,0,:,int(2*self.hparams.DT/3+1):],0,1)).reshape(outputs.shape[2],outputs.shape[0]*(outputs.shape[3]-int(2*self.hparams.DT/3+1)))
            Y1 = (torch.moveaxis(targets_GT[:,0,:,int(2*self.hparams.DT/3+1)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            X1 = (torch.moveaxis(outputs[:,0,:,int(2*self.hparams.DT/3+1)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            loss_KL1 = max(K_hat_tensor(m_seuil[:,0],X1,Y1)-K_hat_tensor(m_seuil[:,0],Y1,Y1),0)
            Y2 = (torch.moveaxis(targets_GT[:,0,:,int(2*self.hparams.DT/3+2)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            X2 = (torch.moveaxis(outputs[:,0,:,int(2*self.hparams.DT/3+2)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            loss_KL2 = max(K_hat_tensor(m_seuil[:,0],X2,Y2)-K_hat_tensor(m_seuil[:,0],Y2,Y2),0)
            Y3 = (torch.moveaxis(targets_GT[:,0,:,int(2*self.hparams.DT/3+3)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
            X3 = (torch.moveaxis(outputs[:,0,:,int(2*self.hparams.DT/3+3)],0,1)).reshape(outputs.shape[2],outputs.shape[0])
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
class CovarianceNetwork(torch.nn.Module):
    def __init__(self, shapeData, dimout, dimH):
        super(CovarianceNetwork,self).__init__()
        self.layer1 = torch.nn.Linear(shapeData, dimH)
        self.layer2 = torch.nn.Linear(dimH, dimout)

    def forward(self, w):
        w = self.layer1(F.relu(w)).to(device)
        w = self.layer2(F.relu(w)).to(device)
        w = (torch.abs(w)).to(device)
        return w
 '''   
    
class LitModelParametric(pl.LightningModule):
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
        self.hparams.iter_update     = [0,10,30,100,150,200,250,400]  # [0,2,4,6,9,15]
        self.hparams.nb_grad_update  = [5,5,10,15,20,20,20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.hparams.lr_update       = [1e-3,1e-4,1e-4,1e-4,1e-4,1e-5,1e-6,1e-6,1e-7]
        self.hparams.k_batch         = 1
        
         
        self.hparams.n_grad          = self.hparams.nb_grad_update[0]
        
        self.hparams.alpha_proj    = 0.5
        self.hparams.alpha_sr      = 0.5
        self.hparams.alpha_lr      = 0.5  # 1e4
        self.hparams.alpha_mse_ssh = 10.
        self.hparams.alpha_mse_gssh = 1.
        self.hparams.alpha_mse_vv = 1
        self.hparams.DT = 13
        self.hparams.w_=np.zeros(self.hparams.DT)
        self.hparams.w_[int(self.hparams.DT / 2)] = 1.
      
        self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(self.hparams.w_), requires_grad=False)
        self.hparams.automatic_optimization = False#True#

        
        self.hparams.NBProjCurrent =self.hparams.NbProjection[0]
        #A revoir
        self.hparams.DimAE=20
        self.hparams.dW = 3
        self.hparams.dW2=1
        self.hparams.sS = 2
        self.hparams.nbBlocks=1
        self.hparams.dropout_phi_r=0
        self.hparams.stochastic=False
        self.hparams.UsePriodicBoundary = False
        self.hparams.dim_grad_solver=150
        self.hparams.dropout =0.25
        
        self.shapeData =[1,31,12]
        self.var_Val = kwargs['var_Val']
        self.var_Tr = kwargs['var_Tr']
        self.var_Tt = kwargs['var_Tt']
        self.meanTrtrue = kwargs['meanTrtrue']
        self.stdTrtrue = kwargs['stdTrtrue']
        #lrCurrent       = lrUpdate[0] ??
        """    
        # main model
        self.model           = NN_4DVar.Model_4DVarNN_GradFP(phi_r,shapeData,self.hparams.NBProjCurrent,self.hparams.n_grad,self.hparams.GradType,self.hparams.OptimType,InterpFlag,UsePriodicBoundary)                
        """
        
        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN_parametric(
            Phi_r(self.shapeData[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS,
                  self.hparams.nbBlocks,self.hparams.dropout_phi_r, self.hparams.stochastic),
            Model_H(self.shapeData[0]),
            CovarianceNetwork(12,3,8),
            NN_4DVar.model_GradUpdateLSTM(self.shapeData, self.hparams.UsePriodicBoundary,
                                          self.hparams.dim_grad_solver, self.hparams.dropout),
            None, None, self.shapeData, self.hparams.n_grad, self.hparams.stochastic)
        
        self.save_hyperparameters()
        self.model_LR = ModelLR()

        self.w_loss       = self.hparams.w_loss # duplicate for automatic upload to gpu
        self.x_pred        = None # variable to store output of test method
        
        self.automatic_optimization = self.hparams.automatic_optimization
        self.curr = 0
        
    def forward(self):
        return 1

    def configure_optimizers(self):
        #optimizer = optim.Adam(self.model.parameters(), lr= self.lrUpdate[0])
        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(),'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': self.hparams.lr_update[0]},
                                  {'params' : self.model.Covariance_model.parameters(), 'lr' : 10*self.hparams.lr_update[0]},
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
            lr = np.array([lrCurrent,lrCurrent,10*lrCurrent])            
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]# * self.hparams.learning_rate
                mm += 1
                
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        opt = self.optimizers()
                    
        # compute loss and metrics    
        loss, out, metrics,Ni,Nreg,LI,Covariance = self.compute_loss(train_batch, phase='train')

        # log step metric        
        #self.log('train_mse', mse)
        #self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] / self.var_Tr , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        loss, out, metrics,Ntot,LI,Covariance,L_AE = self.compute_loss(val_batch, phase='val')
        print('Cov:')
        print(Covariance[:2,0,0,9:])
        self.log('val_loss', torch.abs(Ntot))
        self.log("val_mse", metrics['mse'] / self.var_Val , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_uv", metrics['mse_uv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics,Ntot,LI,Covariance,L_AE = self.compute_loss(test_batch, phase='test')

        self.log('test_loss', loss)
        self.log("test_mse", metrics['mse'] / self.var_Tt , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("test_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("test_vv", metrics['mse_vv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        out_debit = out
        Cov_debit = Covariance
        return {'preds_debit': out_debit.detach().cpu(), 'Covariance' : Cov_debit.detach().cpu()}

    def training_epoch_end(self, training_step_outputs):
        opt = self.optimizers()
        print(opt)
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
        x_test_rec = self.stdTrtrue * x_test_rec + self.meanTrtrue
        self.x_rec_debit = x_test_rec[:,0,:,:]
        Cov_test_rec = torch.cat([chunk['Covariance']for chunk in outputs]).numpy()
        self.Cov_debit = Cov_test_rec[:,0,:,:]
        print("shapetest")
        print(self.x_rec_debit.shape)
        print(debit_gt.shape)
        print(debit_obs.shape)
        path_save1 = self.logger.log_dir + '/test.nc'
        save_NetCDFparam(path_save1, 
                        Indtest,
                        debit_gt = debit_gt , 
                        debit_obs = debit_obs,
                        debit_rec = self.x_rec_debit,
                        Cov_rec = self.Cov_debit,
                        mode='prevision'
                    )
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

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, masks)
            Cov=torch.zeros(inputs_init.shape)+1
            Cov = Cov.to(device)
            Cov[:,:,:,9:] = self.model.Covariance_model(torch.cat((inputs_init[:,:,:,:9],outputs[:,:,:,9:]),3))
            
            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()
                Cov = Cov.detach()
            
            loss_R      = torch.sum((outputs - targets_GT)**2 * masks )
            loss_R      = torch.mul(1.0 / torch.sum(masks),loss_R)
            loss_I      = torch.sum((outputs - targets_GT)**2 * (1. - masks) )
            loss_I      = torch.mul(1.0 / torch.sum(1.-masks),loss_I)
            loss_Abs    = torch.abs(torch.sum((outputs - targets_GT) * (1. - masks) ,0))
            loss_Abs    = torch.sum(loss_Abs)
            loss_Abs    = torch.mul(1.0 / torch.sum(1.-masks),loss_Abs)
            loss_All    = torch.mean((outputs - targets_GT)**2 )
            loss_AE     = torch.mean((self.model.phi_r(outputs) - outputs)**2 )
            loss_AE_GT  = torch.mean((self.model.phi_r(targets_GT) - targets_GT)**2 )
            New_loss_reg = -2*torch.sum(torch.log(Cov))
            New_loss_I  = torch.sum(Cov*(outputs-targets_GT)**2*(1. - masks))
            New_loss_tot  = torch.mul(1.0 / torch.sum(1.-masks),(New_loss_I+New_loss_reg))
            
            if (phase == 'val') or (phase == 'test'):
                print('Ntot:')
                print(New_loss_tot)
                print('Li:')
                print(loss_I)
                print('loss_AE')
                print(loss_AE)
                print('Loss_abs')
                print(loss_Abs)
            
            
            #total loss without reanalysis
            loss = self.hparams.alpha[0] * torch.abs(New_loss_tot) + self.hparams.alpha[1] * ( loss_AE + loss_AE_GT )+ loss_Abs
            
            # metrics
            mse = New_loss_tot.detach()
            metrics   = dict([('mse',mse)])
            #print(mse.cpu().detach().numpy())
            
            
            
            outputs = outputs
        return loss,outputs, metrics, New_loss_tot,loss_I,Cov,loss_AE  
    
    

class GPDNetwork(torch.nn.Module):
    def __init__(self, shapeData, dimout, dimH):
        super(CovarianceNetwork,self).__init__()
        self.layer1 = torch.nn.Linear(shapeData, dimH)
        self.layer2 = torch.nn.Linear(dimH, dimout)

    def forward(self, w):
        w = self.layer1(F.relu(w)).to(device)
        w = self.layer2(F.relu(w)).to(device)
        w = (w**2).to(device)
        return w    
    
    
    
class LitModelParametricGPD(pl.LightningModule):
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
        self.hparams.alpha_mse_vv = 1
        self.hparams.DT = 13
        self.hparams.w_=np.zeros(self.hparams.DT)
        self.hparams.w_[int(self.hparams.DT / 2)] = 1.
      
        self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(self.hparams.w_), requires_grad=False)
        self.hparams.automatic_optimization = False#True#

        
        self.hparams.NBProjCurrent =self.hparams.NbProjection[0]
        #A revoir
        self.hparams.DimAE=20
        self.hparams.dW = 3
        self.hparams.dW2=1
        self.hparams.sS = 2
        self.hparams.nbBlocks=1
        self.hparams.dropout_phi_r=0
        self.hparams.stochastic=False
        self.hparams.UsePriodicBoundary = False
        self.hparams.dim_grad_solver=150
        self.hparams.dropout =0.25
        
        self.shapeData =[1,31,12]
        self.var_Val = kwargs['var_Val']
        self.var_Tr = kwargs['var_Tr']
        self.var_Tt = kwargs['var_Tt']
        self.meanTrtrue = kwargs['meanTrtrue']
        self.stdTrtrue = kwargs['stdTrtrue']
        #lrCurrent       = lrUpdate[0] ??
        """    
        # main model
        self.model           = NN_4DVar.Model_4DVarNN_GradFP(phi_r,shapeData,self.hparams.NBProjCurrent,self.hparams.n_grad,self.hparams.GradType,self.hparams.OptimType,InterpFlag,UsePriodicBoundary)                
        """
        
        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN_parametric(
            Phi_r(self.shapeData[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS,
                  self.hparams.nbBlocks,self.hparams.dropout_phi_r, self.hparams.stochastic),
            Model_H(self.shapeData[0]),
            CovarianceNetwork(12,9,6),
            NN_4DVar.model_GradUpdateLSTM(self.shapeData, self.hparams.UsePriodicBoundary,
                                          self.hparams.dim_grad_solver, self.hparams.dropout),
            None, None, self.shapeData, self.hparams.n_grad, self.hparams.stochastic)
        
        self.save_hyperparameters()
        self.model_LR = ModelLR()

        self.w_loss       = self.hparams.w_loss # duplicate for automatic upload to gpu
        self.x_pred        = None # variable to store output of test method
        
        self.automatic_optimization = self.hparams.automatic_optimization
        self.curr = 0
        
    def forward(self):
        return 1

    def configure_optimizers(self):
        #optimizer = optim.Adam(self.model.parameters(), lr= self.lrUpdate[0])
        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(),'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params' : self.model.Covariance_model.parameters(), 'lr' : self.hparams.lr_update[0]}
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
        loss, out, metrics,Ntot,LI,L_AE, mu,sig,ksi = self.compute_loss(train_batch, phase='train')

        # log step metric        
        #self.log('train_mse', mse)
        #self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] / self.var_Tr , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        
        loss, out, metrics,Ntot,LI,L_AE, mu,sig,ksi =  self.compute_loss(val_batch, phase='val')
        print(loss)
        self.log('val_loss', loss)
        #self.log("val_mse", metrics['mse'] / self.var_Val , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_uv", metrics['mse_uv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics,Ntot,LI,L_AE, mu,sig,ksi  = self.compute_loss(test_batch, phase='test')

        self.log('test_loss', loss)
        self.log("test_mse", metrics['mse'] / self.var_Tt , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("test_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("test_vv", metrics['mse_vv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        out_debit = out
        return {'preds_debit': out_debit.detach().cpu(), 'Mu' : mu.detach().cpu(), 'Sigma' : sig.detach().cpu(), 'Xi' : ksi.detach().cpu()}

    def training_epoch_end(self, training_step_outputs):
        opt = self.optimizers()
        print(opt)
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
        x_test_rec = self.stdTrtrue * x_test_rec + self.meanTrtrue
        self.x_rec_debit = x_test_rec[:,0,:,:]
        Mu = torch.cat([chunk['Mu']for chunk in outputs]).numpy()
        self.Mu_debit = Mu[:,0,:,:]
        Sigma = torch.cat([chunk['Sigma']for chunk in outputs]).numpy()
        self.Sigma_debit = Sigma[:,0,:,:]
        Xi = torch.cat([chunk['Xi']for chunk in outputs]).numpy()
        self.Xi_debit = Xi[:,0,:,:]
        print("shapetest")
        print(self.x_rec_debit.shape)
        print(debit_gt.shape)
        print(debit_obs.shape)
        path_save1 = self.logger.log_dir + '/test.nc'
        save_NetCDFparamGPD(path_save1, 
                        Indtest,
                        debit_gt = debit_gt , 
                        debit_obs = debit_obs,
                        debit_rec = self.x_rec_debit,
                        mu = self.Mu_debit,
                        sigma = self.Sigma_debit,
                        xi = self.Xi_debit,
                        mode='prevision'
                    )
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

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, masks)
            Ksi=torch.zeros(inputs_init.shape)+1
            Ksi = Ksi.to(device)
            sigma=torch.zeros(inputs_init.shape)+1
            sigma = sigma.to(device)
            mu=torch.zeros(inputs_init.shape)+1
            mu = mu.to(device)
            b = torch.zeros(inputs_init.shape)+1
            borne_inf= 0.00001*b
            borne_inf = borne_inf.to(device)
            C=self.model.Covariance_model(torch.cat((inputs_init[:,:,:,:9],outputs[:,:,:,9:]),3))
            Ksi[:,:,:,9:]=C[:,:,:,:3]
            sigma[:,:,:,9:]=C[:,:,:,3:6]
            mu[:,:,:,9:] = C[:,:,:,6:]
            
            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()
                Ksi =Ksi.detach()
                sigma = sigma.detach()
                mu = mu.detach()
            
            
            loss_R      = torch.sum((outputs - targets_GT)**2 * masks )
            loss_R      = torch.mul(1.0 / torch.sum(masks),loss_R)
            loss_I      = torch.sum((outputs - targets_GT)**2 * (1. - masks) )
            loss_I      = torch.mul(1.0 / torch.sum(1.-masks),loss_I)
            loss_All    = torch.mean((outputs - targets_GT)**2 )
            loss_AE     = torch.mean((self.model.phi_r(outputs) - outputs)**2 )
            loss_AE_GT  = torch.mean((self.model.phi_r(targets_GT) - targets_GT)**2 )
            New_loss_reg = torch.sum((torch.log(sigma)*(1.-masks)))
            New_loss_I  = torch.sum((1+1/Ksi)*torch.log(torch.maximum(1+Ksi*(targets_GT-mu)/sigma,borne_inf))*(1.-masks))
            New_loss_tot  = torch.abs(torch.mul(1.0 / torch.sum(1.-masks),(New_loss_I+New_loss_reg)))
            
            if (phase == 'val') or (phase == 'test'): 
                print('Ntot:')
                print(New_loss_tot)
                print('NI')
                print(New_loss_I)
                print('Nreg')
                print(New_loss_reg)
                print('Li:')
                print(loss_I)
                print('loss_AE')
                print(loss_AE)
                print('mu')
                print(mu[0,0,0,9:])
                print('sigma')
                print(sigma[0,0,0,9:])
                print('ksi')
                print(Ksi[0,0,0,9:])
            
            #total loss without reanalysis
            loss = self.hparams.alpha[0] * torch.abs(New_loss_tot) + 0.5 * self.hparams.alpha[1] * ( loss_AE + loss_AE_GT )+0.1*loss_I
            
            # metrics
            mse = New_loss_tot.detach()
            metrics   = dict([('mse',mse)])
            #print(mse.cpu().detach().numpy())
            
            
            
            outputs = outputs
        return loss,outputs, metrics,New_loss_tot,loss_I,loss_AE, mu,sigma, Ksi  
    
    
   
class LitModelParametricInside(pl.LightningModule):
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
        self.hparams.alpha_mse_vv = 1
        self.hparams.DT = 13
        self.hparams.w_=np.zeros(self.hparams.DT)
        self.hparams.w_[int(self.hparams.DT / 2)] = 1.
      
        self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(self.hparams.w_), requires_grad=False)
        self.hparams.automatic_optimization = False#True#

        
        self.hparams.NBProjCurrent =self.hparams.NbProjection[0]
        #A revoir
        self.hparams.DimAE=20
        self.hparams.dW = 3
        self.hparams.dW2=1
        self.hparams.sS = 2
        self.hparams.nbBlocks=1
        self.hparams.dropout_phi_r=0
        self.hparams.stochastic=False
        self.hparams.UsePriodicBoundary = False
        self.hparams.dim_grad_solver=150
        self.hparams.dropout =0.25
        
        self.shapeData =[1,31,12]
        self.var_Val = kwargs['var_Val']
        self.var_Tr = kwargs['var_Tr']
        self.var_Tt = kwargs['var_Tt']
        self.meanTrtrue = kwargs['meanTrtrue']
        self.stdTrtrue = kwargs['stdTrtrue']
        self.b_inf = torch.from_numpy(-self.meanTrtrue/self.stdTrtrue)
        print(self.b_inf.shape)
        #lrCurrent       = lrUpdate[0] ??
        """    
        # main model
        self.model           = NN_4DVar.Model_4DVarNN_GradFP(phi_r,shapeData,self.hparams.NBProjCurrent,self.hparams.n_grad,self.hparams.GradType,self.hparams.OptimType,InterpFlag,UsePriodicBoundary)                
        """
        # main model
        self.model = NN_4DVar.Solver_Grad_4DVarNN_inside(
            Phi_r_inside(self.shapeData[0], self.hparams.DimAE, self.hparams.dW, self.hparams.dW2, self.hparams.sS,
                  self.hparams.nbBlocks,self.hparams.dropout_phi_r,self.b_inf, self.hparams.stochastic),
            Model_H(self.shapeData[0]),
            NN_4DVar.model_GradUpdateLSTM(self.shapeData, self.hparams.UsePriodicBoundary,
                                          self.hparams.dim_grad_solver, self.hparams.dropout),
            None, None, self.shapeData, self.hparams.n_grad, self.hparams.stochastic)
        
        self.save_hyperparameters()
        self.model_LR = ModelLR()

        self.w_loss       = self.hparams.w_loss # duplicate for automatic upload to gpu
        self.x_pred        = None # variable to store output of test method
        
        self.automatic_optimization = self.hparams.automatic_optimization
        self.curr = 0
        
    def forward(self):
        return 1

    def configure_optimizers(self):
        #optimizer = optim.Adam(self.model.parameters(), lr= self.lrUpdate[0])
        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(),'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': 0.1*self.hparams.lr_update[0]},
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
        loss, out, metrics,loss_I,loss_AE, mu,Ksi, sigma = self.compute_loss(train_batch, phase='train')

        # log step metric        
        #self.log('train_mse', mse)
        #self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mse", metrics['mse'] / self.var_Tr , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        
        loss,outputs, metrics,loss_I,loss_AE, mu,Ksi, sigma =  self.compute_loss(val_batch, phase='val')
        print(loss)
        self.log('val_loss', loss)
        #self.log("val_mse", metrics['mse'] / self.var_Val , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_uv", metrics['mse_uv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def test_step(self, test_batch, batch_idx):
        loss,outputs, metrics,loss_I,loss_AE, mu,Ksi, sigma  = self.compute_loss(test_batch, phase='test')

        self.log('test_loss', loss)
        self.log("test_mse", metrics['mse'] / self.var_Tt , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("test_mseG", metrics['mse_grad'] / metrics['meanGrad'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("test_vv", metrics['mse_vv']  , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        out_debit = outputs
        mu        = mu
        sig       = sigma
        ksi       = Ksi
        print('Xi')
        print(Ksi[0,0,:,0])
        print('Sigma')
        print(sig[0,0,:,0])
        return {'preds_debit': out_debit.detach().cpu(), 'Mu' : mu.detach().cpu()}

    def training_epoch_end(self, training_step_outputs):
        opt = self.optimizers()
        print(opt)
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
        x_test_rec = self.stdTrtrue * x_test_rec + self.meanTrtrue
        self.x_rec_debit = x_test_rec[:,0,:,:]
        Mu = torch.cat([chunk['Mu']for chunk in outputs]).numpy()
        self.Mu_debit = Mu[:,0,:,:]
        
        print("shapetest")
        print(self.x_rec_debit.shape)
        print(debit_gt.shape)
        print(debit_obs.shape)
        path_save1 = self.logger.log_dir + '/test.nc'
        save_NetCDFparam(path_save1, 
                        Indtest,
                        debit_gt = debit_gt , 
                        debit_obs = debit_obs,
                        debit_rec = self.x_rec_debit,
                        Cov_rec = self.Mu_debit,
                        
                        mode='prevision'
                    )
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

            outputs, hidden_new, cell_new, normgrad = self.model(inputs_init, inputs_missing, masks)
            phi_GT = self.model.phi_r(targets_GT)[:,:,:,:12]
            out_param = self.model.phi_r(outputs)            
            mu = out_param[:,:,:,:12] 
            Ksi = out_param[:,:,:,12]
            Ksi = torch.unsqueeze(Ksi,dim=3)+0.0001
            Ksi = Ksi.expand(-1,-1,-1,12) 
            sigma = out_param[:,:,:,13]+0.0001
            sigma = torch.unsqueeze(sigma,dim=3)
            sigma = sigma.expand(-1,-1,-1,12) 
            ind_pos = torch.where((outputs-mu)>=0)
            ind_neg = torch.where((outputs-mu)<0)
            ind_pos_GT = torch.where((targets_GT-phi_GT)>=0)
            ind_neg_GT = torch.where((targets_GT-phi_GT)<0)
            
            
            if (phase == 'val') or (phase == 'test'):
                outputs = outputs.detach()
                Ksi =Ksi.detach()
                sigma = sigma.detach()
                mu = mu.detach()            
            
            loss_R      = torch.sum((outputs - targets_GT)**2 * masks )
            loss_R      = torch.mul(1.0 / torch.sum(masks),loss_R)
           
            loss_I      = torch.sum((outputs - targets_GT)**2 * (1.-masks))
            loss_I      = torch.mul(1.0 / torch.sum(1.-masks),loss_I)
            
           
            #loss for the dynamical
            
            loss_AE     = torch.sum((torch.log(sigma)))+torch.sum((1+1/Ksi[ind_pos])*torch.log(1+Ksi[ind_pos]*((outputs-mu)[ind_pos])/sigma[ind_pos]))+torch.sum((1+1/Ksi[ind_neg])*((mu-outputs)[ind_neg]+0.5))
            loss_AE = torch.abs(torch.mul(1.0 / (torch.sum(1.-masks)+torch.sum(masks)),loss_AE))
            loss_AE_GT  = torch.sum((torch.log(sigma)))+torch.sum((1+1/Ksi[ind_pos_GT])*torch.log(1+Ksi[ind_pos_GT]*((targets_GT[ind_pos_GT]-phi_GT[ind_pos_GT]))/sigma[ind_pos_GT]))+torch.sum((1+1/Ksi[ind_neg_GT])*((phi_GT[ind_neg_GT]-targets_GT[ind_neg_GT])+0.5))
            loss_AE_GT = torch.abs(torch.mul(1.0 / (torch.sum(1.-masks)+torch.sum(masks)),loss_AE_GT))
            
            if (phase == 'val') or (phase == 'test'): 
                print('Xi')
                print(Ksi[0,0,0,0])
                print("mu")
                print(mu[0,0,0,0])
                print(sigma[0,0,0,0])
                print("loss_AE")
                print(loss_AE)
                print("loss_I")
                print(loss_I)
            
            #total loss without reanalysis
            loss = self.hparams.alpha[0] * torch.abs(loss_I)  + 0.5 * self.hparams.alpha[0] * ( loss_AE + loss_AE_GT )+0.1*loss_R
            
            # metrics
            mse = loss.detach()
            metrics   = dict([('mse',mse)])
            #print(mse.cpu().detach().numpy())
            
            
        return loss,outputs, metrics,loss_I,loss_AE, mu,Ksi, sigma      
    