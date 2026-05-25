import binascii
from functools import partial
import glob
import hashlib
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool
import random
import copy
from socket import timeout

import numpy as np
import torch
from rdkit.Chem import MolToSmiles, MolFromSmiles, AddHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.transforms import BaseTransform
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from lightning.pytorch import LightningDataModule

from .process_mols import read_molecule, get_rec_graph, generate_conformer, \
	get_lig_graph_with_matching, extract_receptor_structure, parse_receptor, parse_pdb_from_path
from .data_utils import modify_conformer, set_time
from ..utils2.utils import read_strings_from_txt
from ..utils.pylogger import RankedLogger

from torch_cluster.radius import radius_graph

log = RankedLogger(__name__, rank_zero_only=True)
X_N = 2000

def t_to_sigma_(t_tr, t_rot, t_tor, args):
    tr_sigma = args.tr_sigma_min * t_tr + args.tr_sigma_max * (1-t_tr)
    rot_sigma = args.rot_sigma_min * t_rot + args.rot_sigma_max * (1-t_rot)
    tor_sigma = args.tor_sigma_min * t_tor + args.tor_sigma_max * (1-t_tor)
    return tr_sigma, rot_sigma, tor_sigma

def sample_uniform():
    x = np.random.randn(3)
    x /= np.linalg.norm(x)
    return x * np.random.uniform(low=-np.pi, high=np.pi)

class NoiseTransform(BaseTransform):
	def __init__(self, t_to_sigma, no_torsion, all_atom):
		self.t_to_sigma = t_to_sigma
		self.tr_sigma_max, self.rot_sigma_max, self.tor_sigma_max = self.t_to_sigma(0, 0, 0)
		self.omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]
		self.SO3_cdf = self.SO3()
		self.no_torsion = no_torsion
		self.all_atom = all_atom

	def __call__(self, data):
		t = np.random.uniform()
		t_tr, t_rot, t_tor = t, t, t
		return self.apply_noise(data, t_tr, t_rot, t_tor)

	def forward(self, data, t=None):
		if t is None:
			t = np.random.uniform()
		else:
			t = float(t)
		t_tr, t_rot, t_tor = t, t, t
		return self.apply_noise(data, t_tr, t_rot, t_tor)

	def SO3(self):
		'''
			For rotation, we randomly perterb x1 to get x0. 
			The rotation vevtor can be obtained in the axis-angle parameterization by sampling a unit vector uniformly
			and random angle omega in [0, pi] according to the following distribution.
			p(w) = (1 - cos w) / pi * f(w), where w = sum_{l=0}^{infty} (2l + 1) exp(-l(l+1) * sigma^2) sin(w(l + 1/2)) / sin(w/2)
		'''
		omegas = np.linspace(0, np.pi, X_N + 1)[1:]
		p = 0
		for l in range(2000):
			p += (2 * l + 1) * np.exp(-l * (l + 1) * self.rot_sigma_max ** 2) * np.sin(omegas * (l + 1 / 2)) / np.sin(omegas / 2)
		density = (1 - np.cos(omegas)) / np.pi * p
		cdf = density.cumsum() / X_N * np.pi
		return cdf

	def sample_SO3_vec(self):
		x = np.random.randn(3)
		x /= np.linalg.norm(x)
		omega = np.interp(np.random.rand(), self.SO3_cdf, self.omegas_array)
		return omega * x

	def apply_noise(self, data, t_tr, t_rot, t_tor, tr_update = None, rot_update=None, torsion_updates=None):
		if not torch.is_tensor(data['ligand'].pos):
			data['ligand'].pos = random.choice(data['ligand'].pos)
		# tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(t_tr, t_rot, t_tor)
		   # set t = 0 since p_0 is the prior
		set_time(data, t_tr, t_rot, t_tor, 1, self.all_atom, device=None)
		# xt = (1 - t) * x0 + t * x1 = (t - 1) (x1 - x0) + x1 = x1 + (1 - t) * update
		# ut = (x1 - xt) / (1 - t) = - update = (x1 - x0)
		tr_update = torch.normal(mean=0, std=self.tr_sigma_max, size=(1, 3)) if tr_update is None else tr_update
		rot_update = torch.tensor(self.sample_SO3_vec(), dtype=torch.float32) if rot_update is None else rot_update
		# rot_update = torch.tensor(sample_uniform(), dtype=torch.float32) if rot_update is None else rot_update
		# rot_update = torch.normal(mean=0, std=self.rot_sigma_max, size=(3, )) if rot_update is None else rot_update

		torsion_updates = np.random.normal(loc=0.0, scale=self.tor_sigma_max, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
		# make torsion_updates in [-pi, pi)
		torsion_updates = (torsion_updates + np.pi) % (2 * np.pi) - np.pi
		torsion_updates = None if self.no_torsion else torsion_updates

		modify_conformer(data, tr_update * (1 - t_tr), 
				   rot_update * (1 - t_rot), 
				   torsion_updates * (1 - t_tor))

		data.u_tr = - tr_update
		data.u_rot = - rot_update.unsqueeze(0)
		data.u_tor = None if self.no_torsion else - torch.from_numpy(torsion_updates).float()
		data.num_torsions = torch.tensor([data['ligand'].edge_mask.sum()], dtype=torch.long)
		return data

class PDBBind(Dataset):
	def __init__(self, root, transform=None, cache_path='data/cache', split_path='data/', limit_complexes=0,
				 receptor_radius=30, num_workers=1, c_alpha_max_neighbors=None, popsize=15, maxiter=15,
				 matching=True, keep_original=False, max_lig_size=None, remove_hs=False, num_conformers=1, all_atoms=False,
				 atom_radius=5, atom_max_neighbors=None, esm_embeddings_path=None, require_ligand=False,
				 protein_path_list=None, ligand_descriptions=None, keep_local_structures=False):

		super(PDBBind, self).__init__(root, transform)
		self.pdbbind_dir = root                             # 'data/PDBBind_processed/'
		self.max_lig_size = max_lig_size                    # None
		self.split_path = split_path                        # 'data/splits/timesplit_no_lig_overlap_train'
		self.limit_complexes = limit_complexes              # 0
		self.receptor_radius = receptor_radius              # 15
		self.num_workers = num_workers 
		self.c_alpha_max_neighbors = c_alpha_max_neighbors  # 24
		self.remove_hs = remove_hs                          # T
		self.esm_embeddings_path = esm_embeddings_path      # 'data/esm2_3billion_embeddings.pt'
		self.require_ligand = require_ligand                # F
		self.protein_path_list = protein_path_list          # F
		self.ligand_descriptions = ligand_descriptions      # None
		self.keep_local_structures = keep_local_structures  # F
		if matching or protein_path_list is not None and ligand_descriptions is not None:
			cache_path += '_torsion'
		if all_atoms:
			cache_path += '_allatoms'
		self.full_cache_path = os.path.join(cache_path, f'limit{self.limit_complexes}'
														f'_INDEX{os.path.splitext(os.path.basename(self.split_path))[0]}'
														f'_maxLigSize{self.max_lig_size}_H{int(not self.remove_hs)}'
														f'_recRad{self.receptor_radius}_recMax{self.c_alpha_max_neighbors}'
											+ ('' if not all_atoms else f'_atomRad{atom_radius}_atomMax{atom_max_neighbors}')
											+ ('' if not matching or num_conformers == 1 else f'_confs{num_conformers}')
											+ ('' if self.esm_embeddings_path is None else f'_esmEmbeddings')
											+ ('' if not keep_local_structures else f'_keptLocalStruct')
											+ ('' if protein_path_list is None or ligand_descriptions is None else str(binascii.crc32(''.join(ligand_descriptions + protein_path_list).encode()))))
		self.popsize, self.maxiter = popsize, maxiter   # 20, 20
		self.matching, self.keep_original = matching, keep_original # T, T
		self.num_conformers = num_conformers                        # 1
		self.all_atoms = all_atoms                                  # F
		self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors     # 5, 8
		# if not os.path.exists(os.path.join(self.full_cache_path, "success.txt")):
		# 	os.makedirs(self.full_cache_path, exist_ok=True)
		# 	if protein_path_list is None or ligand_descriptions is None:
		# 		self.preprocessing()        # T
		
		# new data load with pkl for each complex data
		self.complexes_names = []
		for file in os.listdir(self.full_cache_path):
			if file.startswith('heterograph_') and file.endswith('.pt'):
				self.complexes_names.append(os.path.join(self.full_cache_path, file))


	def len(self):
		return len(self.complexes_names)

	def get(self, idx):
		file = self.complexes_names[idx]
		path = os.path.join(self.full_cache_path, file)
		data = torch.load(path, weights_only=False)
		if isinstance(data['ligand'].mask_rotate, torch.Tensor):
			data['ligand'].mask_rotate = data['ligand'].mask_rotate.numpy()

		# cdist = torch.cdist(data['receptor'].pos.float(), data['ligand'].pos.float())
		# cdist_min, _ = cdist.min(dim=1)
		# mask = cdist_min < 25

		# # 2. Map original receptor indices to compact indices after cropping.
		# num_nodes_old = data['receptor'].pos.size(0)
		# node_indices = torch.arange(num_nodes_old, device=mask.device)
		# kept_indices = node_indices[mask]  # Original indices of receptor nodes kept

		# # index_mapping[old_idx] = new idx; dropped nodes remain -1
		# index_mapping = torch.full((num_nodes_old,), -1, device=mask.device, dtype=torch.long)
		# index_mapping[kept_indices] = torch.arange(kept_indices.size(0), device=mask.device)

		# # 3. Filter rec_contact edges to the cropped receptor subgraph
		# edge_index_old = data['receptor', 'rec_contact', 'receptor'].edge_index

		# # 3.1 Keep edges where both endpoints survive the spatial mask
		# src_mask = mask[edge_index_old[0]]
		# dst_mask = mask[edge_index_old[1]]
		# valid_edge_mask = src_mask & dst_mask

		# # 3.2 Relabel endpoints with the compact index mapping
		# edge_index_filtered = edge_index_old[:, valid_edge_mask]
		# edge_index_new = index_mapping[edge_index_filtered]

		# data['receptor'].pos = data['receptor'].pos[mask]
		# data['receptor'].x = data['receptor'].x[mask]
		# data['receptor'].mu_r_norm = data['receptor'].mu_r_norm[mask]
		# data['receptor'].side_chain_vecs= data['receptor'].side_chain_vecs[mask]

		# data['receptor', 'rec_contact', 'receptor'].edge_index = edge_index_new
		return data

	def preprocessing(self):
		log.info(f'Processing complexes from [{self.split_path}] and saving it to [{self.full_cache_path}]')

		complex_names_all = read_strings_from_txt(self.split_path)
		if self.limit_complexes is not None and self.limit_complexes != 0:
			complex_names_all = complex_names_all[:self.limit_complexes]
		log.info(f'Loading {len(complex_names_all)} complexes.')

		if self.esm_embeddings_path is not None:
			id_to_embeddings = torch.load(self.esm_embeddings_path)
			chain_embeddings_dictlist = defaultdict(list)
			for key, embedding in id_to_embeddings.items():
				key_name = key.split('_')[0]
				if key_name in complex_names_all:
					chain_embeddings_dictlist[key_name].append(embedding)
			lm_embeddings_chains_all = []
			for name in complex_names_all:
				lm_embeddings_chains_all.append(chain_embeddings_dictlist[name])
		else:
			lm_embeddings_chains_all = [None] * len(complex_names_all)
		log.info(f'Starting preprocess.')
		if self.num_workers > 1:
			freq = 3000
			# running preprocessing in parallel on multiple workers and saving the progress every 3000 complexes
			for i in range(len(complex_names_all)//freq+1):
				complex_names = complex_names_all[freq*i:freq*(i+1)]
				if os.path.exists(os.path.join(self.full_cache_path, f"heterograph_{complex_names[-1]}.pt")) and \
					os.path.exists(os.path.join(self.full_cache_path, f"rdkit_ligand_{complex_names[-1]}.pt")):
					continue
				lm_embeddings_chains = lm_embeddings_chains_all[freq*i:freq*(i+1)]
				complex_graphs, rdkit_ligands = [], []
				if self.num_workers > 1:
					p = Pool(self.num_workers, maxtasksperchild=1)
					p.__enter__()
				with tqdm(total=len(complex_names), desc=f'loading complexes {i}/{len(complex_names_all)//freq+1}') as pbar:
					map_fn = p.imap_unordered if self.num_workers > 1 else map
					for t in map_fn(self.get_complex, zip(complex_names, lm_embeddings_chains, [None] * len(complex_names), [None] * len(complex_names))):
						complex_graphs.extend(t[0])
						rdkit_ligands.extend(t[1])
						pbar.update()
				if self.num_workers > 1: 
					p.__exit__(None, None, None)

				for complex_graph, rdkit_ligand in zip(complex_graphs, rdkit_ligands):
					name = complex_graph['name']
					pro_name = os.path.join(self.full_cache_path, f'heterograph_{name}.pt')
					lig_name = os.path.join(self.full_cache_path, f'rdkit_ligand_{name}.pt')
					if os.path.exists(pro_name) and os.path.exists(lig_name):
						continue
					torch.save(complex_graph, pro_name)
					torch.save(rdkit_ligand, lig_name)
			with open(os.path.join(self.full_cache_path, "success.txt"), 'w') as f:
				f.write('success all')
		else:
			complex_graphs, rdkit_ligands = [], []
			with tqdm(total=len(complex_names_all), desc='loading complexes') as pbar:
				for t in map(self.get_complex, zip(complex_names_all, lm_embeddings_chains_all, [None] * len(complex_names_all), [None] * len(complex_names_all))):
					complex_graphs.extend(t[0])
					rdkit_ligands.extend(t[1])
					pbar.update()
			for complex_graph, rdkit_ligand in zip(complex_graphs, rdkit_ligands):
				name = complex_graph['name']
				pro_name = os.path.join(self.full_cache_path, f'heterograph_{name}.pt')
				lig_name = os.path.join(self.full_cache_path, f'rdkit_ligand_{name}.pt')
				if os.path.exists(pro_name) and os.path.exists(lig_name):
					continue
				torch.save(complex_graph, pro_name)
				torch.save(rdkit_ligand, lig_name)

	def get_complex(self, par):
		name, lm_embedding_chains, ligand, ligand_description = par
		if not os.path.exists(os.path.join(self.pdbbind_dir, name)) and ligand is None:
			log.info("Folder not found", name)
			return [], []

		if ligand is not None:
			rec_model = parse_pdb_from_path(name)
			name = f'{name}____{ligand_description}'
			ligs = [ligand]
		else:
			# rec_model = parse_receptor(name, self.pdbbind_dir)
			try:
				rec_model = parse_receptor(name, self.pdbbind_dir)
			except Exception as e:
				log.info(f'Skipping {name} because of the error:')
				log.info(e)
				return [], []

			ligs = read_mols(self.pdbbind_dir, name, remove_hs=False)
		complex_graphs = []
		failed_indices = []
		for i, lig in enumerate(ligs):
			if self.max_lig_size is not None and lig.GetNumHeavyAtoms() > self.max_lig_size:
				log.info(f'Ligand with {lig.GetNumHeavyAtoms()} heavy atoms is larger than max_lig_size {self.max_lig_size}. Not including {name} in preprocessed data.')
				continue
			complex_graph = HeteroData()
			complex_graph['name'] = name
			try:
				get_lig_graph_with_matching(lig, complex_graph, self.popsize, self.maxiter, self.matching, self.keep_original,
											self.num_conformers, remove_hs=self.remove_hs)
				rec, rec_coords, c_alpha_coords, n_coords, c_coords, lm_embeddings = extract_receptor_structure(copy.deepcopy(rec_model), lig, lm_embedding_chains=lm_embedding_chains)
				if lm_embeddings is not None and len(c_alpha_coords) != len(lm_embeddings):
					log.info(f'LM embeddings for complex {name} did not have the right length for the protein. Skipping {name}.')
					failed_indices.append(i)
					continue

				get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, complex_graph, rec_radius=self.receptor_radius,
							  c_alpha_max_neighbors=self.c_alpha_max_neighbors, all_atoms=self.all_atoms,
							  atom_radius=self.atom_radius, atom_max_neighbors=self.atom_max_neighbors, remove_hs=self.remove_hs, lm_embeddings=lm_embeddings)

			except Exception as e:
				log.info(f'Skipping {name} because of the error:')
				log.info(e)
				failed_indices.append(i)
				continue

			protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
			complex_graph['receptor'].pos -= protein_center
			if self.all_atoms:
				complex_graph['atom'].pos -= protein_center

			if (not self.matching) or self.num_conformers == 1:
				complex_graph['ligand'].pos -= protein_center
			else:
				for p in complex_graph['ligand'].pos:
					p -= protein_center

			complex_graph.original_center = protein_center
			complex_graphs.append(complex_graph)
		for idx_to_delete in sorted(failed_indices, reverse=True):
			del ligs[idx_to_delete]
		return complex_graphs, ligs

class MergeDataset(Dataset):
	def __init__(self, datasets):
		super(MergeDataset, self).__init__()
		self.datasets = datasets
		self.transform = datasets[0].transform
	
	def len(self):
		return sum([dataset.len() for dataset in self.datasets])
	
	def get(self, idx):
		for dataset in self.datasets:
			if idx < dataset.len():
				return dataset.get(idx)
			else:
				idx -= dataset.len()
	
def print_statistics(complex_graphs):
	statistics = ([], [], [], [])

	for complex_graph in complex_graphs:
		lig_pos = complex_graph['ligand'].pos if torch.is_tensor(complex_graph['ligand'].pos) else complex_graph['ligand'].pos[0]
		radius_protein = torch.max(torch.linalg.vector_norm(complex_graph['receptor'].pos, dim=1))
		molecule_center = torch.mean(lig_pos, dim=0)
		radius_molecule = torch.max(
			torch.linalg.vector_norm(lig_pos - molecule_center.unsqueeze(0), dim=1))
		distance_center = torch.linalg.vector_norm(molecule_center)
		statistics[0].append(radius_protein)
		statistics[1].append(radius_molecule)
		statistics[2].append(distance_center)
		if "rmsd_matching" in complex_graph:
			statistics[3].append(complex_graph.rmsd_matching)
		else:
			statistics[3].append(0)

	name = ['radius protein', 'radius molecule', 'distance protein-mol', 'rmsd matching']
	log.info('Number of complexes: ', len(complex_graphs))
	for i in range(4):
		array = np.asarray(statistics[i])
		log.info(f"{name[i]}: mean {np.mean(array)}, std {np.std(array)}, max {np.max(array)}")


def construct_dataset(args, t_to_sigma):
	transform = NoiseTransform(no_torsion=args.no_torsion,
							   all_atom=args.all_atoms, t_to_sigma=t_to_sigma)

	common_args = {'transform': transform, 'root': args.data_dir, 'limit_complexes': args.limit_complexes,
				   'receptor_radius': args.receptor_radius,
				   'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
				   'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
				   'matching': not args.no_torsion, 'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
				   'num_workers': args.num_workers_for_preprocess, 'all_atoms': args.all_atoms,
				   'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
				   'esm_embeddings_path': args.esm_embeddings_path}

	train_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
	 						num_conformers=args.num_conformers, **common_args)
	val_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_val, keep_original=True, **common_args)
	return train_dataset, val_dataset

def read_mol(pdbbind_dir, name, remove_hs=False):
	lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.sdf'), remove_hs=remove_hs, sanitize=True)
	if lig is None:  # read mol2 file if sdf file cannot be sanitized
		log.info('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
		lig = read_molecule(os.path.join(pdbbind_dir, name, f'{name}_ligand.mol2'), remove_hs=remove_hs, sanitize=True)
	return lig

def read_mols(pdbbind_dir, name, remove_hs=False):
	ligs = []
	for file in os.listdir(os.path.join(pdbbind_dir, name)):
		if file.endswith(".sdf") and 'rdkit' not in file:
			lig = read_molecule(os.path.join(pdbbind_dir, name, file), remove_hs=remove_hs, sanitize=True)
			if lig is None and os.path.exists(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2")):  # read mol2 file if sdf file cannot be sanitized
				log.info('Using the .sdf file failed. We found a .mol2 file instead and are trying to use that.')
				lig = read_molecule(os.path.join(pdbbind_dir, name, file[:-4] + ".mol2"), remove_hs=remove_hs, sanitize=True)
			if lig is not None:
				ligs.append(lig)
	return ligs

class PDBBindDataModule(LightningDataModule):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.t_to_sigma = partial(t_to_sigma_, args=args)
		self.train_dataset, self.valid_dataset = construct_dataset(args, self.t_to_sigma)

	def setup(self, stage):
		return

	def train_dataloader(self):
		train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size_per_device, num_workers=self.args.num_workers, shuffle=True)
		return train_loader

	def val_dataloader(self):
		val_loader = DataLoader(self.valid_dataset, batch_size=self.args.batch_size_per_device, num_workers=self.args.num_workers, shuffle=False)
		return val_loader
	
