
Clone the repo Github
Create a dedicated conda environment:
conda create -n 4dvarnetsto mamba pytorch=1.11 torchvision cudatoolkit=11.3 -c conda-forge -c pytorch
conda activate 4dvarnetsto
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

Other required packages:
netcdf4
