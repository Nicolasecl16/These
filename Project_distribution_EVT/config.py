params = {
    'data_dir'        : '/gpfsscratch/rech/nlu/commun/large',
    'dir_save'        : '/gpfsscratch/rech/nlu/commun/large/results_maxime',

    'iter_update'     : [0, 20, 40, 60, 100, 150, 800],  # [0,2,4,6,9,15]
    'nb_grad_update'  : [15, 10, 10, 10, 15, 15, 20, 20, 20],#[5, 5, 10, 10, 15, 15, 20, 20, 20],  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
    'lr_update'       : [1e-4, 1e-4, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7],#[1e-3, 1e-4, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7],
    'k_batch'         : 1,
    'n_grad'          : 5,
    'dT'              : 10, ## Time window of each space-time patch
    'dx'              : 1,   ## subsampling step if > 1
    'W'               : 200, # width/height of each space-time patch
    'resize_factor'   : 2,
    'dW'              : 3,
    'dW2'             : 1,
    'sS'              : 4,  # int(5/dx), # !! must be compatible with original_size/resize_factor
    'nbBlocks'        : 1,
    'Nbpatches'       : 1, #10#10#25 ## number of patches extracted from each time-step 
    'NbStations'      : 31,

    # stochastic version
    'stochastic'      : False,
    
    #parametric version 
    'parametric'      : False,
    'GPD'             : False,
    'inside'          : True,

    # animation maps 
    'animate'         : False,

    # NN architectures and optimization parameters
    'batch_size'      : 64, #16#4#4#8#12#8#256#
    'DimAE'           : 50, #10#10#50
    'dim_grad_solver' : 150,
    'dropout'         : 0.25,
    'dropout_phi_r'   : 0.,

    'alpha_proj'      : 0.5,
    'alpha_sr'        : 0.5,
    'alpha_lr'        : 0.5,  # 1e4
    'alpha_mse_ssh'   : 10.,
    'alpha_mse_gssh'  : 1.,

    # data generation
    'sigNoise'        : 0.,## additive noise standard deviation
    'flagSWOTData'    : True, #False ## use SWOT data or not
    'Nbpatches'       : 1, #10#10#25 ## number of patches extracted from each time-step 
    'rnd1'            : 0, ## random seed for patch extraction (space sam)
    'rnd2'            : 100, ## random seed for patch extraction
    'dwscale'         : 1,

    'UsePriodicBoundary' : False,  # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
    'InterpFlag'         : False, # True :> force reconstructed field to observed data after each gradient-based update
    'automatic_optimization' : True,

}
