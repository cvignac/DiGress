# DiGress: Discrete Denoising diffusion models for graph generation (ICLR 2023)


Warning: The code has been updated after experiments were run for the paper. If you don't manage to reproduce the 
paper results, please write to us so that we can investigate the issue.

For the conditional generation experiments, check the `guidance` branch. 

## Environment installation
  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit: `conda create -c conda-forge -n digress rdkit python=3.9`
  - Install graph-tool (https://graph-tool.skewed.de/): `conda install -c conda-forge graph-tool`
  - Install the nvcc drivers for your cuda version. For example, `conda install -c "nvidia/label/cuda-11.3.1" cuda-nvcc`
  - Install pytorch 1.10 or 1.11 (https://pytorch.org/)
  - Install pytorch-geometric. Your version should match the pytorch version that is installed (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  - Install other packages using the requirement file: `pip install -r requirements.txt`
  - Install mini-moses: `pip install git+https://github.com/igor-krawczuk/mini-moses`
  - Run `pip install -e .`
  - Navigate to the ./util/orca directory and compile orca.cpp: `g++ -O2 -std=c++11 -o orca orca.cpp`


## Download the data

  - QM9 and Guacamol should download by themselves when you run the code.
  - For the community, SBM and planar datasets, data can be found at https://github.com/KarolisMart/SPECTRE/tree/main/data
  - Moses dataset can be found at https://github.com/molecularsets/moses/tree/master/data
  


## Run the code
  
  - All code is currently launched through `python3 main.py`. Check hydra documentation (https://hydra.cc/) for overriding default parameters.
  - To run the debugging code: `python3 main.py +experiment=debug.yaml`. We advise to try to run the debug mode first
    before launching full experiments.
  - To run a code on only a few batches: `python3 main.py general.name=test`.
  - To run the continuous model: `python3 main.py model=continuous`
  - To run the discrete model: `python3 main.py`
  - You can specify the dataset with `python3 main.py dataset=guacamol`. Look at `configs/dataset` for the list
of datasets that are currently available
    
## Checkpoints

NOTE: since the code reformatting, these commits cannot be loaded anymore. If you want to use them, either use the commit `682e59019dd33073b1f0f4d3aaba7de6a308602e` or rename `src` to `dgd`, and then run `pip install -e .`

We uploaded pretrained models for the Planar and SBM datasets. If you need other checkpoints, please write to us.

Planar: https://drive.switch.ch/index.php/s/tZCjJ6VXU2Z3FIh
SBM: https://drive.switch.ch/index.php/s/rxWFVQX4Cu4Vq5j

## Generated samples

We provide the generated samples for some of the models. If you have retrained a model from scratch for which the samples are
not available yet, we would be very happy if you could send them to us!
    
## Cite the paper

```
@inproceedings{
vignac2023digress,
title={DiGress: Discrete Denoising diffusion for graph generation},
author={Clement Vignac and Igor Krawczuk and Antoine Siraudin and Bohan Wang and Volkan Cevher and Pascal Frossard},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=UaAD-Nu86WX}
}
```
