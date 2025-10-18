# [NeurIPS 2025] ForceFM: Enhancing Protein-Ligand Predictions through Force-Guided Flow Matching
![Alt Text](overview.png)

The detailed code will be released soon. 


# Dataset

The files in `data` contain the names for the time-based data split.

If you want to train one of our models with the data then: 
1. download it from [zenodo](https://zenodo.org/record/6408497) 
2. unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`



## Setup Environment

We will set up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html). Clone the current repo

    git clone https://github.com/Guhuary/ForceFM.git

This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, pytorch-geometric, cuda versions or cpu only versions):

    conda create --name diffdock python=3.8
    conda activate diffdock
    conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas

Then you need to install ESM that we use both for protein sequence embeddings and for the protein structure prediction in case you only have the sequence of your target. Note that OpenFold (and so ESMFold) requires a GPU. If you don't have a GPU, you can still use DiffDock with existing protein structures.

    pip install "fair-esm[esmfold]"
    pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
    pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'


## Citation
    @inproceedings{guo2025forcefm,
    title={ForceFM: Enhancing Protein-Ligand Predictions through Force-Guided Flow Matching},
    author={Huanlei Guo, Song Liu and Bingyi Jing},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=e7HEbUVryj}
    }

## License
MIT




