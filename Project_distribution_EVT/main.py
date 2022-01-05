#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:59:23 2020
@author: rfablet
"""
import os

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint


import Updated_Solver as NN_4DVar
#from lit_model_stochastic import LitModelStochastic
from models import Gradient_img, LitModel
#from new_dataloading import FourDVarNetDataModule
import new_dataloading as Data
#from old_dataloading import LegacyDataLoading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gradient_img = Gradient_img()


class FourDVarNetRunner:
    def __init__(self, dataloading="old", config=None):
        self.filename_chkpt = 'modelsum-Exp2-{epoch:02d}-{val_loss:.2f}'
        if config is None:
            import config 
        else:
            config = __import__("config_" + str(config))

        self.cfg = OmegaConf.create(config.params)
        
        self.dataloaders = Data.dataloaders
        self.std_Tr = Data.std_Tr_obs.cpu().detach().numpy()
        self.mean_Tr = Data.mean_Tr_obs.cpu().detach().numpy()

        
        self.lit_cls = LitModel
        
            
    def run(self, ckpt_path=None, dataloader="test", **trainer_kwargs):
        """
        Train and test model and run the test suite
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param dataloader: Dataloader on which to run the test Checkpoint from which to resume
        :param trainer_kwargs: (Optional)
        """
        mod, trainer = self.train(ckpt_path, **trainer_kwargs)
        self.test(dataloader=dataloader, _mod=mod, _trainer=trainer)

    def _get_model(self, ckpt_path=None):
        """
        Load model from ckpt_path or instantiate new model
        :param ckpt_path: (Optional) Checkpoint path to load
        :return: lightning module
        """

        if ckpt_path:
            mod = self.lit_cls.load_from_checkpoint(ckpt_path, 
                                                    mean_Tr=self.mean_Tr, 
                                                    std_Tr = self.std_Tr
                                                    )
        else:
            mod = self.lit_cls(hparam=self.cfg, 
                               mean_Tr=self.mean_Tr, std_Tr= self.std_Tr
                              )
        return mod

    def train(self, ckpt_path='Res_Update/modelsum-Exp2-epoch=99-val_loss=-7.38.ckpt', **trainer_kwargs):
        """
        Train a model
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param trainer_kwargs: (Optional) Trainer arguments
        :return:
        """

        mod = self._get_model(ckpt_path=ckpt_path)
        print(mod.model)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                              dirpath = './Res_Update/',
                                              filename=self.filename_chkpt,
                                              save_top_k=3,
                                              mode='min')
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        num_gpus = torch.cuda.device_count()
        accelerator = "ddp" if (num_gpus * num_nodes) > 1 else None
        trainer = pl.Trainer(num_nodes=num_nodes, gpus=num_gpus, accelerator=accelerator, auto_select_gpus=True,
                             callbacks=[checkpoint_callback], max_epochs = 1000, **trainer_kwargs)
        
        x_init,obs,mask,_ = next(iter(self.dataloaders['train']))
        print("x_init")
        print(x_init.shape)
        #mod.model(x_init,obs,mask)
        trainer.fit(mod, self.dataloaders['train'], self.dataloaders['val'])
        
        return mod, trainer

    
    def test(self, ckpt_path='Res_Update/modelsum-Exp2-epoch=90-val_loss=-4.60.ckpt', dataloader="test", _mod=None, _trainer=None, **trainer_kwargs):
        """
        Test a model
        :param ckpt_path: (Optional) Checkpoint from which to resume
        :param dataloader: Dataloader on which to run the test Checkpoint from which to resume
        :param trainer_kwargs: (Optional)
        """

        mod = _mod or self._get_model(ckpt_path=ckpt_path)

        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
        num_gpus = torch.cuda.device_count()
        accelerator = "ddp" if (num_gpus * num_nodes) > 1 else None
        # trainer = _trainer or pl.Trainer(num_nodes=num_nodes, gpus=num_gpus, accelerator=accelerator, **trainer_kwargs)
        trainer = pl.Trainer(num_nodes=1, gpus=1, accelerator=None, **trainer_kwargs)
        print(mod)
        trainer.test(mod, test_dataloaders=self.dataloaders[dataloader])

    def profile(self):
        """
        Run the profiling
        :return:
        """
        from pytorch_lightning.profiler import PyTorchProfiler

        profiler = PyTorchProfiler(
            "results/profile_report",
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=1),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./tb_profile'),
            record_shapes=True,
            profile_memory=True,
        )
        self.train(
            **{
                'profiler': profiler,
                'max_epochs': 1,
            }
        )


if __name__ == '__main__':
    import fire

    fire.Fire(FourDVarNetRunner)
