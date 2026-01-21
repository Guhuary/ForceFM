
import copy
import os
import torch
from argparse import ArgumentParser, Namespace
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader

from src.datasets.process_mols import write_mol_with_coords
from src.utils2.inference_utils import InferenceDataset, set_nones
from src.utils2.sampling import randomize_position, sampling_fm
# from src.utils2.utils import get_model, get_diffdock_confidence_model
from src.utils2.visualise import PDBFile
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*') # type: ignore
import yaml

import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument('--protein_ligand_csv', type=str, default='/mnt/sharedata/ssd_large/users/guohl/datasets/ai4sci/pdbbind2020/testset_csv.csv', help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
parser.add_argument('--complex_name', type=str, default='1a0q', help='Name that the complex will be saved with')
parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')
parser.add_argument('--esm_embeddings_path', default='/mnt/sharedata/ssd_large/users/guohl/datasets/ai4sci/pdbbind2020/esm2_3billion_embeddings.pt', type=str, help='Path to the ESM embeddings. If this is None, no ESM embeddings will be used')
parser.add_argument('--out_dir', type=str, default='results/finetune', help='Directory where the outputs will be written to')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
parser.add_argument('--samples_per_complex', type=int, default=40, help='Number of samples to generate')

parser.add_argument('--model_dir', type=str, default='workdir/basemodel', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='epoch_1939.ckpt', help='Checkpoint to use for the score model')
parser.add_argument('--confidence_model_dir', type=str, default='workdir/diffdock_confidence_model', help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')

parser.add_argument('--batch_size', type=int, default=5, help='')
parser.add_argument('--inference_steps', type=int, default=11, help='Number of denoising steps')
parser.add_argument('--actual_steps', type=int, default=10, help='Number of denoising steps that are actually performed')
parser.add_argument('--seed', type=int, default=42, help='seed')
args = parser.parse_known_args()[0]

os.makedirs(args.out_dir, exist_ok=True)
# from src.utils.safe_yaml_loader import SkipPyTagsLoader
from omegaconf import OmegaConf
conf = OmegaConf.load(f'{args.model_dir}/hparams.yaml')
# fm_model_args = conf.args
fm_model_args = OmegaConf.merge(conf.args, conf.data.args)
if args.confidence_model_dir is not None:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num = 400
if args.protein_ligand_csv is not None:
    df = pd.read_csv(args.protein_ligand_csv)
    complex_name_list = set_nones(df['complex_name'].tolist())[:num]
    protein_path_list = set_nones(df['protein_path'].tolist())[:num]
    protein_path_list = [os.path.join('/mnt/sharedata/ssd_large/users/guohl/datasets/ai4sci/pdbbind2020', p[5:]) for p in protein_path_list] # type: ignore
    protein_sequence_list = set_nones(df['protein_sequence'].tolist())[:num]
    ligand_description_list = set_nones(df['ligand_description'].tolist())[:num]
    ligand_description_list = [os.path.join('/mnt/sharedata/ssd_large/users/guohl/datasets/ai4sci/pdbbind2020', p[5:]) for p in ligand_description_list] # type: ignore
else:
    complex_name_list = [args.complex_name]
    protein_path_list = [args.protein_path]
    protein_sequence_list = [args.protein_sequence]
    ligand_description_list = [args.ligand_description]


# complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
for name in complex_name_list:
    write_dir = f'{args.out_dir}/{name}'
    os.makedirs(write_dir, exist_ok=True)


# preprocessing of complexes into geometric graphs
test_dataset = InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                lm_embeddings=args.esm_embeddings_path is not None,
                                receptor_radius=fm_model_args.receptor_radius, remove_hs=fm_model_args.remove_hs, # type: ignore
                                c_alpha_max_neighbors=fm_model_args.c_alpha_max_neighbors, # type: ignore
                                all_atoms=fm_model_args.all_atoms, atom_radius=fm_model_args.atom_radius, # type: ignore
                                atom_max_neighbors=fm_model_args.atom_max_neighbors) # type: ignore
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


if args.confidence_model_dir is not None and not confidence_args.use_original_model_cache:
    print('HAPPENING | confidence model uses different type of graphs than the score model. '
          'Loading (or creating if not existing) the data for the confidence model now.')
    confidence_test_dataset = \
        InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                         ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                         lm_embeddings=args.esm_embeddings_path is not None,
                         receptor_radius=confidence_args.receptor_radius, remove_hs=confidence_args.remove_hs,
                         c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                         all_atoms=confidence_args.all_atoms, atom_radius=confidence_args.atom_radius,
                         atom_max_neighbors=confidence_args.atom_max_neighbors,
                         precomputed_lm_embeddings=test_dataset.lm_embeddings)
else:
    confidence_test_dataset = None

def t_to_sigma_conf(t_tr, t_rot, t_tor, tr_sigma_min=0.1, tr_sigma_max=19, rot_sigma_min=0.03, rot_sigma_max=1.55, tor_sigma_min=0.0314, tor_sigma_max=3.14):
    tr_sigma = tr_sigma_min ** (1-t_tr) * tr_sigma_max ** t_tr
    rot_sigma = rot_sigma_min ** (1-t_rot) * rot_sigma_max ** t_rot
    tor_sigma = tor_sigma_min ** (1-t_tor) * tor_sigma_max ** t_tor
    return tr_sigma, rot_sigma, tor_sigma

def get_diffdock_confidence_model(args, t_to_sigma=t_to_sigma_conf, confidence_mode=True):
    from src.models.time_step_embedding import get_timestep_embedding
    from src.models.all_atom_score_model import TensorProductScoreModel as ConfidenceModel

    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type,
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale)

    lm_embedding_type = None
    if args.esm_embeddings_path is not None: 
        lm_embedding_type = 'esm'

    model = ConfidenceModel(t_to_sigma=t_to_sigma,
                        no_torsion=args.no_torsion,
                        timestep_emb_func=timestep_emb_func,
                        num_conv_layers=args.num_conv_layers,
                        lig_max_radius=args.max_radius,
                        scale_by_sigma=args.scale_by_sigma,
                        sigma_embed_dim=args.sigma_embed_dim,
                        ns=args.ns, nv=args.nv,
                        distance_embed_dim=args.distance_embed_dim,
                        cross_distance_embed_dim=args.cross_distance_embed_dim,
                        batch_norm=not args.no_batch_norm,
                        dropout=args.dropout,
                        use_second_order_repr=args.use_second_order_repr,
                        cross_max_distance=args.cross_max_distance,
                        dynamic_max_cross=args.dynamic_max_cross,
                        lm_embedding_type=lm_embedding_type, # type: ignore
                        confidence_mode=confidence_mode,
                        num_confidence_outputs=len(
                            args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                            args.rmsd_classification_cutoff, list) else 1)
    return model

from src.module.FlowMatch import Base_FM_Model
flow_model = Base_FM_Model.load_from_checkpoint(f'{args.model_dir}/{args.ckpt}')
model = copy.deepcopy(flow_model.ema).eval()

if args.confidence_model_dir is not None:
    confidence_model = get_diffdock_confidence_model(confidence_args, t_to_sigma=t_to_sigma_conf, confidence_mode=True)
    state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
    confidence_model.load_state_dict(state_dict, strict=True)
    confidence_model = confidence_model.to(device)
    confidence_model.eval()
else:
    confidence_model = None
    confidence_args = None


t_schedule = np.linspace(0, 1, args.inference_steps + 1)[:-1]

failures, skipped = 0, 0
N = args.samples_per_complex    # 40
print('Size of test dataset: ', len(test_dataset))  # 2
print(args)

for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
    if not orig_complex_graph.success[0]:
        skipped += 1
        print(f"HAPPENING | The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
        continue
    if True:
        if os.path.exists(os.path.join(f'{args.out_dir}/{complex_name_list[idx]}', 'rank1.sdf')):
            print(complex_name_list[idx], 'exists')
            continue
        if confidence_test_dataset is not None:
            confidence_complex_graph = confidence_test_dataset[idx]
            if not confidence_complex_graph.success: # type: ignore
                skipped += 1
                print(f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
                continue
            confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
        else:
            confidence_data_list = None
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        randomize_position(data_list, fm_model_args.no_torsion, False, fm_model_args.tr_sigma_max)    # X, F, F, 19.0
        lig = orig_complex_graph.mol[0]

        # initialize visualisation
        pdb = None
        if args.save_visualisation:
            visualization_list = []
            for graph in data_list:
                pdb = PDBFile(lig)
                pdb.add(lig, 0, 0)
                pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                visualization_list.append(pdb)
        else:
            visualization_list = None

        # run reverse diffusion
        data_list, confidence = sampling_fm(data_list=data_list, model=model,
                                         inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                         tr_schedule=t_schedule, rot_schedule=t_schedule, tor_schedule=t_schedule,
                                         device=device, model_args=fm_model_args,
                                         visualization_list=visualization_list, confidence_model=confidence_model,
                                         confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                                         batch_size=args.batch_size)
        ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])

        # reorder predictions based on confidence output
        if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list): # type: ignore
            confidence = confidence[:, 0]
        if confidence is not None:
            confidence = confidence.cpu().numpy()
            re_order = np.argsort(confidence)[::-1]
            confidence = confidence[re_order]
            ligand_pos = ligand_pos[re_order]

        # save predictions
        write_dir = f'{args.out_dir}/{complex_name_list[idx]}'
        for rank, pos in enumerate(ligand_pos):
            mol_pred = copy.deepcopy(lig)
            if fm_model_args.remove_hs: 
                mol_pred = RemoveHs(mol_pred)
            if rank == 0: 
                write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
            write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf')) # type: ignore

        # save visualisation frames
        if args.save_visualisation:
            if confidence is not None:
                for rank, batch_idx in enumerate(re_order):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb')) # type: ignore
            else:
                for rank, batch_idx in enumerate(ligand_pos):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb')) # type: ignore
    
    # except Exception as e:
    #     torch.cuda.empty_cache()
    #     print("Failed on", orig_complex_graph["name"], e)
    #     failures += 1
    
print(f'Failed for {failures} complexes')
print(f'Skipped {skipped} complexes')
print(f'Results are in {args.out_dir}')





