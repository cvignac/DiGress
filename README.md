# DiGress: Discrete Denoising diffusion models for graph generation

Update (July 11th, 2023): the code now supports multi-gpu. Please update all libraries according to the instructions. 
All datasets should now download automatically

  - For the conditional generation experiments, check the `guidance` branch.
  - If you are training new models from scratch, we recommand to use the `fixed_bug` branch in which some neural
network layers have been fixed. The `fixed_bug` branch has not been evaluated, but should normally perform better.
If you train the `fixed_bug` branch on datasets provided in this code, we would be happy to know the results.

## Environment installation
This code was tested with PyTorch 2.0.1, cuda 11.8 and torch_geometrics 2.3.1

  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit:
    
    ```conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9```
  - `conda activate digress`
  - Check that this line does not return an error:
    
    ``` python3 -c 'from rdkit import Chem' ```
  - Install graph-tool (https://graph-tool.skewed.de/): 
    
    ```conda install -c conda-forge graph-tool=2.45```
  - Check that this line does not return an error:
    
    ```python3 -c 'import graph_tool as gt' ```
  - Install the nvcc drivers for your cuda version. For example:
    
    ```conda install -c "nvidia/label/cuda-11.8.0" cuda```
  - Install a corresponding version of pytorch, for example: 
    
    ```pip3 install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118```
  - Install other packages using the requirement file: 
    
    ```pip install -r requirements.txt```

  - Run:
    
    ```pip install -e .```

  - Navigate to the ./src/analysis/orca directory and compile orca.cpp: 
    
     ```g++ -O2 -std=c++11 -o orca orca.cpp```


## Download the data

  - QM9 and Guacamol should download by themselves when you run the code.
  - For the community, SBM and planar datasets, data can be found at https://github.com/KarolisMart/SPECTRE/tree/main/data
     - For SBM, you can use: `wget https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt`
     - For planar, `wget https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt`
    Download the files and simply place them in the `data` folder.
  - Moses dataset can be found at https://github.com/molecularsets/moses/tree/master/data
  
If you want to run Guacamol on the filtered data, either download it from https://drive.switch.ch/index.php/s/pjlZ8A7PADiBGrr
or follow these instructions:
  - Set filter_dataset=True in `guacamol_dataset.py`
  - Run main.py with cfg.dataset.filtered=False
  - Delete data/guacamol/guacamol_pyg/processed
  - Run main.py with cfg.dataset.filtered=True

Note: graph_tool and torch_geometric currently seem to conflict on MacOS, I have not solved this issue yet.

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

The following checkpoints should work with the latest commit:

  - Planar: https://drive.switch.ch/index.php/s/hRWLp8gOGOGFzgR \\
    Performance of this checkpoint: 
    - Test NLL: 1135.6080 
    - `{'spectre': 0.006211824145982536, 'clustering': 0.0563302653184386, 'orbit': 0.00980205113753696, 'planar_acc': 0.85, 'sampling/frac_unique': 1.0, 'sampling/frac_unique_non_iso': 1.0, 'sampling/frac_unic_non_iso_valid': 0.85, 'sampling/frac_non_iso': 1.0} `

  - MOSES (the model in the paper was trained a bit longer than this one): https://drive.switch.ch/index.php/s/DBbvfMmezjg6KUm \\
    Performance of this checkpoint:
    - Test NLL: 203.8171 
    - `{'valid': 0.86032, 'unique@1000': 1.0, 'unique@10000': 0.9999, 'FCD/Test': 0.6176261401223826, 'SNN/Test': 0.5493580505032953, 'Frag/Test': 0.9986637035374839, 'Scaf/Test': 0.8997144919185305, 'FCD/TestSF': 1.2799741890619032, 'SNN/TestSF': 0.5231424506655995, 'Frag/TestSF': 0.9968362360368359, 'Scaf/TestSF': 0.11830576038721641, 'IntDiv': 0.8550915438149056, 'IntDiv2': 0.8489191659624407, 'Filters': 0.9707550678817184, 'logP': 0.02719348046624242, 'SA': 0.05725088257521343, 'QED': 0.0043940205061221965, 'weight': 0.7913020095007184, 'Novelty': 0.9442790697674419}`

  - SBM: https://drive.switch.ch/index.php/s/rxWFVQX4Cu4Vq5j \\
    Performance of this checkpoint:
    - Test NLL: 4757.903
    - `{'spectre': 0.0060240439382095445, 'clustering': 0.05020166160905111, 'orbit': 0.04615866844490847, 'sbm_acc': 0.675, 'sampling/frac_unique': 1.0, 'sampling/frac_unique_non_iso': 1.0, 'sampling/frac_unic_non_iso_valid': 0.625, 'sampling/frac_non_iso': 1.0}`


The following checkpoints require to revert to commit `682e59019dd33073b1f0f4d3aaba7de6a308602e` and run `pip install -e .`:

  - Guacamol: https://drive.switch.ch/index.php/s/jjM3pIHdxWrUGOH

## Generated samples

We provide the generated samples for some of the models. If you have retrained a model from scratch for which the samples are
not available yet, we would be very happy if you could send them to us!


## Troubleshooting 

`PermissionError: [Errno 13] Permission denied: '/home/vignac/DiGress/src/analysis/orca/orca'`: You probably did not compile orca.
    

## Use DiGress on a new dataset

To implement a new dataset, you will need to create a new file in the `src/datasets` folder. Depending on whether you are considering
molecules or abstract graphs, you can base this file on `moses_dataset.py` or `spectre_datasets.py`, for example. 
This file should implement a `Dataset` class to process the data (check [PyG documentation](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_dataset.html)), 
as well as a `DatasetInfos` class that is used to define the noise model and some metrics.

For molecular datasets, you'll need to specify several things in the DatasetInfos:
  - The atom_encoder, which defines the one-hot encoding of the atom types in your dataset
  - The atom_decoder, which is simply the inverse mapping of the atom encoder
  - The atomic weight for each atom atype
  - The most common valency for each atom type

The node counts and the distribution of node types and edge types can be computed automatically using functions from `AbstractDataModule`.

Once the dataset file is written, the code in main.py can be adapted to handle the new dataset, and a new file can be added in `configs/dataset`.


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
