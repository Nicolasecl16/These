# 4DVarnet-Sto expriments

## Environement set-up
- Clone the repo Github with the command git clone
- Create a dedicated conda environment:
```
conda create -n 4dvarnetsto mamba pytorch=1.11 torchvision cudatoolkit=11.3 -c conda-forge -c pytorch
conda activate 4dvarnetsto
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

- Other required packages:
```
netcdf4
pytorch-lightning=1.6.2
```

## Train a model
In one of the folder Danube or Lorenz63
From a terminal
```
python main.py --train
```

## Test pre-trained model
In one of the folder Danube or Lorenz63, open Analysis.ipynb for detailed experiments.
