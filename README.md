# [NeurIPS 2025] ForceFM: Enhancing Protein-Ligand Predictions through Force-Guided Flow Matching
![Alt Text](overview.png)



# I‘m preparing for CVPR 2026. The detailed code will be released in two weeks. 

# Dataset

The files in `data` contain the names for the time-based data split. If you want to train one of our models with the data then: 

1. download it from [zenodo](https://zenodo.org/record/6408497) used in [DiffDock](https://github.com/gcorso/DiffDock)

2. unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`

Each entry is named by a unique identifier (e.g., PDB ID), and follows the structure below:

```
la0q/
├── la0q_ligand.mol2
├── la0q_ligand.sdf
└── la0q_protein_processed.pdb
```

## Setup Environment

Important dependencies: pytorch, lightning, hydra, torch-scatter, torch-sparse, torch-cluster, torch-geometric, fair-esm[esmfold], esm, rdkit

## Retraining ForceFM

Download the data and place it as described in the "Dataset" section above.

### Generate the ESM2 embeddings for the proteins

First run:

```bash
python src/datasets/pdbbind_lm_embedding_preparation.py
```

Use the generated file `data/pdbbind_sequences.fasta` to generate the ESM2 language model embeddings using the library https://github.com/facebookresearch/esm by installing their repository and executing the following in their repository:

```bash
python ${path_to_esm_folder}/scripts/extract.py esm2_t33_650M_UR50D data/pdbbind_sequences.fasta data/embeddings_output --repr_layers 33 --include per_tok --truncation_seq_length 4096
```

This generates the `embeddings_output` directory which you have to copy into the `data` folder of our repository to have `data/embeddings_output`.

Then run the command:

```bash
python src/datasets/esm_embeddings_to_pt.py
```

### **Using the provided model weights for evaluation**

We first generate the language model embeddings for the testset, then run inference with ForceFM, and then evaluate the files that produced:

```bash
python src/datasets/esm_embedding_preparation.py --protein_ligand_csv data/testset_csv.csv --out_file data/prepared_for_esm_testset.fasta
python ${path_to_esm_folder}/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm_testset.fasta data/esm2_output --repr_layers 33 --include per_tok
```




## Citation
    @inproceedings{guo2025forcefm,
    	title={ForceFM: Enhancing Protein-Ligand Predictions through Force-Guided Flow Matching},
    	author={Huanlei Guo, Song Liu and Bingyi Jing},
    	booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    	year={2025},
    	url={https://openreview.net/forum?id=e7HEbUVryj}
    }





