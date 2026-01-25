# 
import os
# os.chdir('../')
from collections import defaultdict, Counter
from multiprocessing import Pool
import random
import copy
from socket import timeout
from argparse import ArgumentParser, Namespace
import re
import subprocess

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from tqdm import tqdm, trange
# from scipy.spatial.transform import Rotation as R
from torch_geometric.transforms import BaseTransform
from src.datasets.process_mols import read_molecule, read_mols, generate_conformer
from src.datasets.data_utils import read_strings_from_txt, set_time, modify_conformer
# from src.datasets.pdbbind import NoiseTransform

complex_names_all = read_strings_from_txt('./data/splits/timesplit_no_lig_overlap_train')

def write_molecule(mol, filepath):
	# Create a molecule copy
	mol = fix_aromatic_kekulize_fail(copy.deepcopy(mol))
	w = Chem.SDWriter(filepath)
	w.write(mol)
	w.close()

def fix_aromatic_kekulize_fail(mol):
	"""When Cannot kekulize, convert all aromatic bonds to single bonds and clear aromatic flags"""
	for bond in mol.GetBonds():
		if bond.GetIsAromatic():
			bond.SetIsAromatic(False)
			bond.SetBondType(Chem.BondType.SINGLE)
	for atom in mol.GetAtoms():
		atom.SetIsAromatic(False)

	# Perform sanitize again, but disable kekulize and set aromaticity steps
	sanitize_flags = (Chem.SanitizeFlags.SANITIZE_ALL 
					  ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE 
					  ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
	Chem.SanitizeMol(mol, sanitizeOps=sanitize_flags)
	return mol

datadir = ''

X_N = 2000
class NoiseTransform(BaseTransform):
	def __init__(self, no_torsion, all_atom, tr_sigma_max=8, rot_sigma_max=1.56, tor_sigma_max=3.14):
		self.tr_sigma_max = tr_sigma_max 
		self.rot_sigma_max = rot_sigma_max 
		self.tor_sigma_max = tor_sigma_max 
		self.omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]
		self.SO3_cdf = self.SO3()
		self.no_torsion = no_torsion
		self.all_atom = all_atom

	def forward(self, data):
		t = np.random.uniform(low=0.9, high=1.0)
		t_tr, t_rot, t_tor = t, t, t
		return self.apply_noise_aug(data, t_tr, t_rot, t_tor)
	
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

	def apply_noise_aug(self, data, t_tr, t_rot, t_tor, tr_update = None, rot_update=None, torsion_updates=None):
		# In real implementation, we sample x0 by adding noise to x1 directlt
		if not torch.is_tensor(data['ligand'].pos):
			data['ligand'].pos = random.choice(data['ligand'].pos)
		# set_time(data, 0, 0, 0, 1, True, device=None)
		tr_update = torch.normal(mean=0, std=self.tr_sigma_max, size=(1, 3)) if tr_update is None else tr_update
		rot_update = self.sample_SO3_vec() if rot_update is None else rot_update

		torsion_updates = np.random.normal(loc=0.0, scale=self.tor_sigma_max, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
		# make torsion_updates in [-pi, pi)
		torsion_updates = (torsion_updates + np.pi) % (2 * np.pi) - np.pi
		torsion_updates = None if self.no_torsion else torsion_updates

		modify_conformer(data, tr_update * (1 - t_tr), 
				   torch.from_numpy(rot_update).float() * (1 - t_rot), 
				   torsion_updates * (1 - t_tor))

		data.tr_update = tr_update * (1 - t_tr)
		data.rot_update = torch.from_numpy(rot_update).float().unsqueeze(0) * (1 - t_rot)
		data.tor_update = None if self.no_torsion else torch.from_numpy(torsion_updates).float().unsqueeze(0) * (1 - t_tor)
		data.num_torsions = torch.tensor([data['ligand'].edge_mask.sum()], dtype=torch.long)
		return data

def _init_noise_transform():
	global _NOISE_TRANSFORM
	_NOISE_TRANSFORM = NoiseTransform(no_torsion=False, all_atom=True)


def _augment_one_complex(name: str):
	"""Generate 40 augmented ligand conformers + save corresponding transforms."""
	try:
		complex_data_path = os.path.join(datadir, f'heterograph_{name}.pt')
		if not os.path.exists(complex_data_path):
			return name, 'skip_no_complex_pt'

		out_dir = f'data/PDBBind_processed/{name}'
		lig_path = os.path.join(out_dir, f'{name}_ligand.sdf')
		if not os.path.exists(lig_path):
			return name, 'skip_no_lig_sdf'

		out_transform = os.path.join(out_dir, f'{name}_aug_transforms.pt')
		# If you want to re-generate even when exists, remove this guard.
		if os.path.exists(out_transform):
			return name, 'skip_exists'

		complex_data = torch.load(complex_data_path, weights_only=False)
		if isinstance(complex_data['ligand'].mask_rotate, torch.Tensor):
			complex_data['ligand'].mask_rotate = complex_data['ligand'].mask_rotate.numpy()

		mol = read_molecule(lig_path, sanitize=False, remove_hs=True)
		conf = mol.GetConformer()

		complex_datas = [
			_NOISE_TRANSFORM.forward(copy.deepcopy(complex_data))
			for _ in range(40)
		]
		pos0 = complex_datas[0]['ligand'].pos
		if len(pos0) != conf.GetNumAtoms():
			return name, 'skip_atom_mismatch'

		tr_updates, rot_updates, tor_updates = [], [], []
		for i in range(len(complex_datas)):
			pos = complex_datas[i]['ligand'].pos + complex_datas[i].original_center
			pos = pos.cpu().numpy().astype(np.float64)
			if len(pos) != conf.GetNumAtoms():
				return name, 'skip_atom_mismatch'
			for j in range(len(pos)):
				x, y, z = pos[j]
				conf.SetAtomPosition(j, Point3D(x, y, z))
			write_molecule(mol, os.path.join(out_dir, f'{name}_aug_{i}.sdf'))
			tr_updates.append(complex_datas[i].tr_update)
			rot_updates.append(complex_datas[i].rot_update)
			tor_updates.append(complex_datas[i].tor_update)

		tr_updates = torch.cat(tr_updates, dim=0)
		rot_updates = torch.cat(rot_updates, dim=0)
		tor_updates = torch.cat(tor_updates, dim=0)
		torch.save(
			{'tr_updates': tr_updates, 'rot_updates': rot_updates, 'tor_updates': tor_updates},
			out_transform,
		)
		return name, 'ok'
	except Exception as e:
		return name, f'error_{repr(e)}'


def _get_nvidia_cuda_version():
	"""Return CUDA Version (major, minor) from nvidia-smi if available, else None."""
	try:
		out = subprocess.check_output(['nvidia-smi'], text=True, stderr=subprocess.STDOUT)
		ml = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
		if not ml:
			return None
		return int(ml.group(1)), int(ml.group(2))
	except Exception:
		return None


def _select_default_gnina_path():
	"""Default: use gnina_cuda if CUDA version >= 12.8 else gnina."""
	v = _get_nvidia_cuda_version()
	if v is not None and v >= (12, 8):
		return '../software/gnina_cuda'
	return '../software/gnina'


def _run_map(names, fn, workers: int, initializer=None):
	"""workers>0 => multiprocessing, workers==0 => sequential map."""
	if workers and workers > 0:
		with Pool(processes=workers, initializer=initializer) as pool:
			for r in pool.imap_unordered(fn, names):
				yield r
	else:
		if initializer is not None:
			initializer()
		for n in names:
			yield fn(n)


complex_names_all = read_strings_from_txt('./data/splits/timesplit_no_lig_overlap_train')

def _parse_args():
	p = ArgumentParser()
	p.add_argument('--workers', type=int, default=0)
	p.add_argument('--datadir', type=str, default='./data/cache_torsion/limit0_INDEXtimesplit_no_lig_overlap_train_maxLigSizeNone_H0_recRad15.0_recMax24_esmEmbeddings')
	p.add_argument('--split', type=str, default='./data/splits/timesplit_no_lig_overlap_train')
	p.add_argument('--gnina_path', type=str, default=None)
	p.add_argument('--stage', type=str, choices=['aug', 'gnina', 'all'], default='all')
	return p.parse_args()

aff_pat = re.compile(r"Affinity:\s+([-\d\.]+)\s+\(kcal/mol\)")


def merge_sdfs(lig_paths, out_path):
	writer = Chem.SDWriter(out_path)
	for p in lig_paths:
		suppl = Chem.SDMolSupplier(p, removeHs=False)
		for mol in suppl:
			if mol is not None:
				writer.write(mol)
	writer.close()


def process_one_complex(name: str):
	"""
	Complete computation logic for a single complex:
	  - Skip if already computed/files incomplete
	  - Merge 40 SDFs
	  - Call gnina
	  - Parse affinity and save pt
	"""
	try:
		out_pt = f'data/PDBBind_processed/{name}/{name}_gnina_affinities.pt'
		if os.path.exists(out_pt):
			# Already computed
			return name, "skip_exists"

		# Check if files are complete
		base_dir = f'data/PDBBind_processed/{name}'
		if not os.path.exists(os.path.join(base_dir, f'{name}_aug_39.sdf')):
			return name, "skip_no_aug39"

		protein_path = os.path.join(base_dir, f'{name}_protein_processed.pdb')
		if not os.path.exists(protein_path):
			return name, "skip_no_protein"

		lig_paths = [
			os.path.join(base_dir, f"{name}_aug_{aug}.sdf")
			for aug in range(40)
		]
		merged_sdf = os.path.join(base_dir, f"{name}_all_augs.sdf")

		# Merge all augments into one SDF
		merge_sdfs(lig_paths, merged_sdf)

		# Remove individual augmented ligand files after merging to save space
		for p in lig_paths:
			try:
				os.remove(p)
			except OSError:
				pass

		# Call gnina
		cmd = [
			gnina_path,
			"-r", protein_path,
			"-l", merged_sdf,
			"--autobox_ligand", merged_sdf,
			"--score_only",
		]
		proc = subprocess.run(
			cmd,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
		)
		out = proc.stdout

		# Parse affinity
		pattern = re.compile(r"Affinity:\s*([-\d\.]+)\s*\(kcal/mol\)")
		affinities = [float(x) for x in pattern.findall(out)]

		# Can also sanity check the number
		if len(affinities) == 0:
			# Indicates gnina might have errored
			# You can also log the out here
			return name, "error_no_affinity"

		torch.save({'affinities': affinities}, out_pt)
		return name, f"ok_{len(affinities)}"

	except Exception as e:
		# Avoid subprocess exception crashing, return error info
		return name, f"error_{repr(e)}"


def main():
	args = _parse_args()
	global datadir, complex_names_all, gnina_path
	
	# Update globals from args
	datadir = args.datadir
	complex_names_all = read_strings_from_txt(args.split)
	gnina_path = args.gnina_path or _select_default_gnina_path()

	print(f'Using gnina: {gnina_path} (workers={args.workers})')

	# Run augmentation stage if requested
	if args.stage in ['aug', 'all']:
		print('\n=== Running augmentation stage ===')
		aug_results = []
		for name, status in tqdm(
			_run_map(complex_names_all, _augment_one_complex, 
			        workers=args.workers, initializer=_init_noise_transform),
			total=len(complex_names_all),
			desc='Augmenting',
		):
			aug_results.append((name, status))

		aug_cnt = Counter(s for _, s in aug_results)
		print('\n[Augmentation]', dict(aug_cnt))

	# Run gnina stage if requested
	if args.stage in ['gnina', 'all']:
		print('\n=== Running gnina scoring stage ===')
		gnina_results = []
		for name, status in tqdm(
			_run_map(complex_names_all, process_one_complex, workers=args.workers),
			total=len(complex_names_all),
			desc='Scoring with gnina',
		):
			gnina_results.append((name, status))

		gnina_cnt = Counter(s for _, s in gnina_results)
		print('\n[Gnina]', dict(gnina_cnt))


if __name__ == "__main__":
	main()
