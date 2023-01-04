# Denoising diffusion models for graph generation


Warning: The paper experiments were run with an old version of the code. We have incorporated the changes into the public
version to create this branch, but we have not tested thoroughly yet. Please tell us if you find any bugs.


This branch contains the code for the guidance model
 - We advise creating a new environment for the guidance model. Follow previous instructions for the installation 
   (you can skip `graph-tool` and `mini-moses`, which are not needed).
 - install Psi4
 - Delete the QM9 dataset -- it needs to be processed again
 - Train a regressor using `python3 guidance/train_qm9_regressor.py +experiment=regressor_model.yaml`
 - Train an unconditional model without extra features, for example: `python3 main.py +experiment=test`
 - In `guidance_homo.yaml` (for example), set the paths of the two checkpoints obtained above
 - Evaluate the guidance model: `python3 guidance/main_guidance.py +experiment=guidance_homo`




## Environment installation
  - Download anaconda/miniconda if needed
  - Create a rdkit environment that directly contains rdkit: `conda create -c conda-forge -n my-rdkit-env rdkit python=3.9`
  - Install pytorch 1.10 or 1.11 (https://pytorch.org/)
  - Install pytorch-geometric. Your version should match the pytorch version that is installed (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  - Install other packages using the requirement file: `pip install -r requirements.txt`




    
## Cite the paper

```
@article{vignac2022digress,
  title={DiGress: Discrete Denoising diffusion for graph generation},
  author={Vignac, Clement and Krawczuk, Igor and Siraudin, Antoine and Wang, Bohan and Cevher, Volkan and Frossard, Pascal},
  journal={arXiv preprint arXiv:2209.14734},
  year={2022}
}
```
