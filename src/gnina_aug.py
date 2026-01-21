# %%
import os
# os.chdir('../')
from collections import defaultdict
from multiprocessing import Pool
import random
import copy
from socket import timeout
from argparse import ArgumentParser, Namespace

import yaml
import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import MolToSmiles, MolFromSmiles, AddHs, RemoveHs
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm, trange
# from scipy.spatial.transform import Rotation as R
from lightning.pytorch import LightningDataModule
from src.datasets.process_mols import read_molecule, read_mols, generate_conformer
from src.datasets.data_utils import read_strings_from_txt, set_time, modify_conformer
datadir = '/mnt/sharedata/ssd_large/users/guohl/datasets/ai4sci/pdbbind2020/cache/limit0_INDEXtimesplit_no_lig_overlap_train_maxLigSizeNone_H0_recRad15.0_recMax24_esmEmbeddings'

# def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
#     if molecule_file.endswith('.mol2'):
#         mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
#     elif molecule_file.endswith('.sdf'):
#         supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=True)
#         mol = supplier[0]
#     elif molecule_file.endswith('.pdbqt'):
#         with open(molecule_file) as file:
#             pdbqt_data = file.readlines()
#         pdb_block = ''
#         for line in pdbqt_data:
#             pdb_block += '{}\n'.format(line[:66])
#         mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
#     elif molecule_file.endswith('.pdb'):
#         mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
#     else:
#         raise ValueError('Expect the format of the molecule_file to be '
#                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

#     try:
#         if sanitize or calc_charges:
#             Chem.SanitizeMol(mol)

#         if calc_charges:
#             # Compute Gasteiger charges on the molecule.
#             try:
#                 AllChem.ComputeGasteigerCharges(mol)
#             except:
#                 warnings.warn('Unable to compute charges for the molecule.')

#         if remove_hs:
#             mol = Chem.RemoveHs(mol, sanitize=sanitize)
#     except Exception as e:
#         print(e)
#         log.info(e)
#         log.info("RDKit was unable to read the molecule.")
#         return None

#     return mol

# X_N = 2000

# class NoiseTransform(BaseTransform):
# 	def __init__(self, no_torsion, all_atom, tr_sigma_max=8, rot_sigma_max=1.56, tor_sigma_max=3.14):
# 		self.tr_sigma_max = tr_sigma_max 
# 		self.rot_sigma_max = rot_sigma_max 
# 		self.tor_sigma_max = tor_sigma_max 
# 		self.omegas_array = np.linspace(0, np.pi, X_N + 1)[1:]
# 		self.SO3_cdf = self.SO3()
# 		self.no_torsion = no_torsion
# 		self.all_atom = all_atom

# 	def forward(self, data):
# 		t = np.random.uniform(low=0.9, high=1.0)
# 		t_tr, t_rot, t_tor = t, t, t
# 		return self.apply_noise_aug(data, t_tr, t_rot, t_tor)
	
# 	def SO3(self):
# 		'''
# 			For rotation, we randomly perterb x1 to get x0. 
# 			The rotation vevtor can be obtained in the axis-angle parameterization by sampling a unit vector uniformly
# 			and random angle omega in [0, pi] according to the following distribution.
# 			p(w) = (1 - cos w) / pi * f(w), where w = sum_{l=0}^{infty} (2l + 1) exp(-l(l+1) * sigma^2) sin(w(l + 1/2)) / sin(w/2)
# 		'''
# 		omegas = np.linspace(0, np.pi, X_N + 1)[1:]
# 		p = 0
# 		for l in range(2000):
# 			p += (2 * l + 1) * np.exp(-l * (l + 1) * self.rot_sigma_max ** 2) * np.sin(omegas * (l + 1 / 2)) / np.sin(omegas / 2)
# 		density = (1 - np.cos(omegas)) / np.pi * p
# 		cdf = density.cumsum() / X_N * np.pi
# 		return cdf

# 	def sample_SO3_vec(self):
# 		x = np.random.randn(3)
# 		x /= np.linalg.norm(x)
# 		omega = np.interp(np.random.rand(), self.SO3_cdf, self.omegas_array)
# 		return omega * x

# 	def apply_noise_aug(self, data, t_tr, t_rot, t_tor, tr_update = None, rot_update=None, torsion_updates=None):
# 		# In real implementation, we sample x0 by adding noise to x1 directlt
# 		if not torch.is_tensor(data['ligand'].pos):
# 			data['ligand'].pos = random.choice(data['ligand'].pos)
# 		# set_time(data, 0, 0, 0, 1, True, device=None)
# 		tr_update = torch.normal(mean=0, std=self.tr_sigma_max, size=(1, 3)) if tr_update is None else tr_update
# 		rot_update = self.sample_SO3_vec() if rot_update is None else rot_update

# 		torsion_updates = np.random.normal(loc=0.0, scale=self.tor_sigma_max, size=data['ligand'].edge_mask.sum()) if torsion_updates is None else torsion_updates
# 		# make torsion_updates in [-pi, pi)
# 		torsion_updates = (torsion_updates + np.pi) % (2 * np.pi) - np.pi
# 		torsion_updates = None if self.no_torsion else torsion_updates

# 		modify_conformer(data, tr_update * (1 - t_tr), 
# 				   torch.from_numpy(rot_update).float() * (1 - t_rot), 
# 				   torsion_updates * (1 - t_tor))

# 		data.tr_update = tr_update * (1 - t_tr)
# 		data.rot_update = torch.from_numpy(rot_update).float().unsqueeze(0) * (1 - t_rot)
# 		data.tor_update = None if self.no_torsion else torch.from_numpy(torsion_updates).float().unsqueeze(0) * (1 - t_tor)
# 		return data

# complex_names_all = read_strings_from_txt('./data/splits/timesplit_no_lig_overlap_train')

# def write_molecule(mol, filepath):
# 	# 创建分子副本 
# 	mol = fix_aromatic_kekulize_fail(copy.deepcopy(mol))
# 	w = Chem.SDWriter(filepath)
# 	w.write(mol)
# 	w.close()

# def fix_aromatic_kekulize_fail(mol):
#     """当 Cannot kekulize 时，把所有芳香键转成单键并清除芳香标记"""
#     for bond in mol.GetBonds():
#         if bond.GetIsAromatic():
#             bond.SetIsAromatic(False)
#             bond.SetBondType(Chem.BondType.SINGLE)
#     for atom in mol.GetAtoms():
#         atom.SetIsAromatic(False)

#     # 再做一次 sanitize，但关闭 kekulize 和设芳香性的步骤
#     sanitize_flags = (Chem.SanitizeFlags.SANITIZE_ALL 
#                       ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE 
#                       ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
#     Chem.SanitizeMol(mol, sanitizeOps=sanitize_flags)
#     return mol

# noise_transform = NoiseTransform(no_torsion=False, all_atom=True)
# for name in tqdm(complex_names_all):
# 	# name == '4o2c'
# 	complex_data_path = os.path.join(datadir, f'{name}.pt')
# 	if not os.path.exists(complex_data_path):
# 		continue

# 	complex_data = torch.load(complex_data_path, weights_only=False)
# 	mol = read_molecule(f'data/PDBBind_processed/{name}/{name}_ligand.sdf', sanitize=False, remove_hs=True)
# 	conf = mol.GetConformer()
# 	complex_datas = [noise_transform.forward(copy.deepcopy(complex_data)) for _ in range(40)]
# 	pos = complex_datas[0]['ligand'].pos
# 	if len(pos) != conf.GetNumAtoms():
# 		continue
# 	tr_updates, rot_updates, tor_updates = [], [], []
# 	for i in range(len(complex_datas)):
# 		pos = complex_datas[i]['ligand'].pos + complex_datas[i].original_center
# 		pos = pos.cpu().numpy().astype(np.float64)
# 		assert len(pos) == conf.GetNumAtoms(), f"{len(pos)} vs {conf.GetNumAtoms()}"
# 		for j in range(len(pos)):
# 			x, y, z = pos[j]
# 			conf.SetAtomPosition(j, Point3D(x, y, z))
# 		write_molecule(mol, f'data/PDBBind_processed/{name}/{name}_aug_{i}.sdf')
# 		tr_updates.append(complex_datas[i].tr_update)
# 		rot_updates.append(complex_datas[i].rot_update)
# 		tor_updates.append(complex_datas[i].tor_update)
# 	tr_updates = torch.cat(tr_updates, dim=0)
# 	rot_updates = torch.cat(rot_updates, dim=0)
# 	tor_updates = torch.cat(tor_updates, dim=0)
# 	# print(tr_updates.shape, rot_updates.shape, tor_updates.shape)
# 	torch.save({'tr_updates': tr_updates, 'rot_updates': rot_updates, 'tor_updates': tor_updates}, 
# 			f'data/PDBBind_processed/{name}/{name}_aug_transforms.pt')
	

# %% [markdown]
# # GNINA Scoring

# %%
# import subprocess, re, os
# complex_names_all = read_strings_from_txt('./data/splits/timesplit_no_lig_overlap_train')
# gnina_path = '/mnt/sharedata/ssd_large/users/guohl/software/gnina_cuda'

# aff_pat = re.compile(r"Affinity:\s+([-\d\.]+)\s+\(kcal/mol\)")
# name_pat = re.compile(r"^##\s+(\S+)")  # 抓每个配体块的名字

# # %%
# def merge_sdfs(lig_paths, out_path):
# 	writer = Chem.SDWriter(out_path)
# 	for p in lig_paths:
# 		suppl = Chem.SDMolSupplier(p, removeHs=False)
# 		for mol in suppl:
# 			if mol is not None:
# 				writer.write(mol)
# 	writer.close()

# # %%
# for name in tqdm(complex_names_all):
# 	if os.path.exists(f'data/PDBBind_processed/{name}/{name}_gnina_affinities.pt'):
# 		continue
# 	if not os.path.exists(f'data/PDBBind_processed/{name}/{name}_aug_39.sdf'):
# 		continue
# 	protein_path = f"data/PDBBind_processed/{name}/{name}_protein_processed.pdb"
# 	if not os.path.exists(protein_path):
# 		continue
# 	lig_paths = [
# 		f"data/PDBBind_processed/{name}/{name}_aug_{aug}.sdf"
# 		for aug in range(40)
# 	]
# 	merged_sdf = f"data/PDBBind_processed/{name}/{name}_all_augs.sdf"
# 	merge_sdfs(lig_paths, merged_sdf)

# 	cmd = [
# 		gnina_path,
# 		"-r", protein_path,
# 		"-l", merged_sdf,
# 		"--autobox_ligand", merged_sdf,
# 		"--score_only",
# 	]
# 	proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
# 	out = proc.stdout

# 	# 解析：按块读，每个块里抓 name + Affinity
# 	pattern = re.compile(r"Affinity:\s*([-\d\.]+)\s*\(kcal/mol\)")
# 	affinities = [float(x) for x in pattern.findall(out)]
# 	torch.save({'affinities': affinities}, f'data/PDBBind_processed/{name}/{name}_gnina_affinities.pt')


import os
import re
import subprocess
from multiprocessing import Pool, cpu_count

import torch
from rdkit import Chem
from tqdm import tqdm

complex_names_all = read_strings_from_txt('./data/splits/timesplit_no_lig_overlap_train')
gnina_path = '/mnt/sharedata/ssd_large/users/guohl/software/gnina_cuda'

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
    单个 complex 的完整计算逻辑：
      - 跳过已经算过/文件不全的
      - merge 40 个 SDF
      - 调 gnina
      - 解析 affinity 并保存 pt
    """
    try:
        out_pt = f'data/PDBBind_processed/{name}/{name}_gnina_affinities.pt'
        if os.path.exists(out_pt):
            # 已经算过
            return name, "skip_exists"

        # 检查文件是否齐全
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

        # 合并所有 augment 到一个 SDF
        merge_sdfs(lig_paths, merged_sdf)

        # 调 gnina
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

        # 解析 affinity
        pattern = re.compile(r"Affinity:\s*([-\d\.]+)\s*\(kcal/mol\)")
        affinities = [float(x) for x in pattern.findall(out)]

        # 可以顺便 sanity check 一下数量
        if len(affinities) == 0:
            # 说明 gnina 可能出错了
            # 你也可以在这里把 out 打到日志里
            return name, "error_no_affinity"

        torch.save({'affinities': affinities}, out_pt)
        return name, f"ok_{len(affinities)}"

    except Exception as e:
        # 避免子进程异常直接挂掉，返回错误信息
        return name, f"error_{repr(e)}"


if __name__ == "__main__":
    # 根据机器情况调节进程数
    # 如果 gnina 用单卡 GPU，可以先试 2~4 个进程，不要一下开很多
    n_workers = 6

    with Pool(processes=n_workers) as pool:
        results = []
        for name, status in tqdm(
            pool.imap_unordered(process_one_complex, complex_names_all),
            total=len(complex_names_all)
        ):
            results.append((name, status))

    # 你可以在这里简单统计一下状态
    from collections import Counter
    cnt = Counter(s for _, s in results)
    print(cnt)
