import math
from functools import partial
from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
import numpy as np
from e3nn.nn import BatchNorm, Activation

# from ..utils2 import so3, torus
from ..datasets.process_mols import lig_feature_dims, rec_residue_feature_dims

from .layers.tensor_product import TensorProductConvLayer
from .layers.common import GaussianSmearing, AtomEncoder, clamped_norm
from ..utils.pylogger import RankedLogger
from .time_step_embedding import get_timestep_embedding

log = RankedLogger(__name__, rank_zero_only=True)

def t_to_sigma(t_tr, t_rot, t_tor, tr_sigma_max, tr_sigma_min, rot_sigma_max, rot_sigma_min, tor_sigma_max, tor_sigma_min):
    tr_sigma = tr_sigma_min * t_tr + tr_sigma_max * (1-t_tr)
    rot_sigma = rot_sigma_min * t_rot + rot_sigma_max * (1-t_rot)
    tor_sigma = tor_sigma_min * t_tor + tor_sigma_max * (1-t_tor)
    return tr_sigma, rot_sigma, tor_sigma

class Tanh_with_pi(nn.Module):
	def __init__(self):
		super(Tanh_with_pi, self).__init__()
	
	def forward(self, x):
		return torch.tanh(x) * math.pi
	
class TensorProductModel(torch.nn.Module):
	def __init__(self, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=1,
				 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
				 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
				 scale_by_sigma=True, smooth_edges: bool = False,
				 dynamic_max_cross=False, dropout=0.0, lm_embedding_type=None, norm_type='layer_norm', 
				 time_step_embedding_type=None, time_step_embedding_scale=100,
				 tr_sigma_max=8, tr_sigma_min=0.1, rot_sigma_max=np.pi / 2, rot_sigma_min=np.pi / 100, tor_sigma_max = np.pi, tor_sigma_min = np.pi / 100, 
				 tanh_out: bool = False): # 
		super(TensorProductModel, self).__init__()
		self.t_to_sigma = partial(t_to_sigma, 
							tr_sigma_max=tr_sigma_max, tr_sigma_min=tr_sigma_min, 
							rot_sigma_max=rot_sigma_max, rot_sigma_min=rot_sigma_min, 
							tor_sigma_max=tor_sigma_max, tor_sigma_min=tor_sigma_min)
		self.in_lig_edge_features = in_lig_edge_features                # 4
		self.sigma_embed_dim = sigma_embed_dim                          # 64
		self.lig_max_radius = lig_max_radius                            # 5
		self.rec_max_radius = rec_max_radius                            # 30
		self.cross_max_distance = cross_max_distance                    # 80
		self.dynamic_max_cross = dynamic_max_cross                      # T
		self.center_max_distance = center_max_distance                  # 30
		self.distance_embed_dim = distance_embed_dim                    # 64
		self.cross_distance_embed_dim = cross_distance_embed_dim        # 64
		self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
		self.ns, self.nv = ns, nv                                       # 48, 10
		self.scale_by_sigma = scale_by_sigma                            # T
		self.no_torsion = no_torsion                                    # F
		self.timestep_emb_func = get_timestep_embedding(
			embedding_type=time_step_embedding_type,
			embedding_dim=sigma_embed_dim,
			embedding_scale=time_step_embedding_scale)
		
		self.num_conv_layers = num_conv_layers                          # 6
		self.activation = nn.ReLU(inplace=True)
		self.smooth_edges = smooth_edges                                # F
		self.norm_type = norm_type                                      # layer_norm

		self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
		self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns), self.activation, nn.Dropout(dropout),nn.Linear(ns, ns))

		self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
		self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), self.activation, nn.Dropout(dropout),nn.Linear(ns, ns))

		self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), self.activation, nn.Dropout(dropout),nn.Linear(ns, ns))

		self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
		self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
		self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

		irrep_seq = [
			f'{ns}x0e',
			f'{ns}x0e + {nv}x1o',
			f'{ns}x0e + {nv}x1o + {nv}x1e',
			f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
		]

		lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
		self.num_conv_layers = num_conv_layers
		for i in range(num_conv_layers):
			in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
			out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
			parameters = {
				'in_irreps': in_irreps,
				'sh_irreps': self.sh_irreps,
				'out_irreps': out_irreps,
				'n_edge_features': 3 * ns,
				'residual': False,
				'dropout': dropout, 
				'hidden_features': 3 * ns,
				'faster': sh_lmax == 1,
				'edge_groups': 1, 
				'tp_weights_layers': 2,
				'activation': 'relu',
				'depthwise': False,
				'norm_type': norm_type,
			}
			lig_layer = TensorProductConvLayer(**parameters)
			lig_conv_layers.append(lig_layer)
			rec_layer = TensorProductConvLayer(**parameters)
			rec_conv_layers.append(rec_layer)
			lig_to_rec_layer = TensorProductConvLayer(**parameters)
			lig_to_rec_conv_layers.append(lig_to_rec_layer)
			rec_to_lig_layer = TensorProductConvLayer(**parameters)
			rec_to_lig_conv_layers.append(rec_to_lig_layer)

		self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
		self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
		self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
		self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)

		# center of mass translation and rotation components
		self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
		self.center_edge_embedding = nn.Sequential(
			nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
			self.activation,
			nn.Dropout(dropout),
			nn.Linear(ns, ns)
		)

		self.final_conv = TensorProductConvLayer(
			in_irreps=self.lig_conv_layers[-1].out_irreps, # type: ignore
			sh_irreps=self.sh_irreps, # type: ignore
			out_irreps=f'2x1o + 2x1e',
			n_edge_features=2 * ns,
			residual=False,
			dropout=dropout,
			norm_type = norm_type # type: ignore
		)

		self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), self.activation, nn.Linear(ns, 1))
		self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), self.activation, nn.Linear(ns, 1), 
			nn.Identity() if not tanh_out else nn.Tanh())

		if not no_torsion:
			# torsion angles components
			self.final_edge_embedding = nn.Sequential(
				nn.Linear(distance_embed_dim, ns),
				self.activation,
				nn.Dropout(dropout),
				nn.Linear(ns, ns)
			)
			self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e") # type: ignore
			self.tor_bond_conv = TensorProductConvLayer(
				in_irreps=self.lig_conv_layers[-1].out_irreps, # type: ignore
				sh_irreps=self.final_tp_tor.irreps_out, # type: ignore
				out_irreps=f'{ns}x0o + {ns}x0e',
				n_edge_features=3 * ns,
				residual=False,
				dropout=dropout,
				norm_type=norm_type, # type: ignore
			)
			self.tor_final_layer = nn.Sequential(
				nn.Linear(2 * ns, ns, bias=False),
				nn.Tanh(),
				nn.Dropout(dropout),
				nn.Linear(ns, 1, bias=False), 
				nn.Identity() if not tanh_out else nn.Tanh()
			)

	def forward(self, data):
		tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])

		# build ligand graph
		lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight = self.build_lig_conv_graph(data)
		lig_src, lig_dst = lig_edge_index
		lig_node_attr = self.lig_node_embedding(lig_node_attr)
		lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

		# build receptor graph
		rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh, rec_edge_weight = self.build_rec_conv_graph(data)
		rec_src, rec_dst = rec_edge_index
		rec_node_attr = self.rec_node_embedding(rec_node_attr)
		rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

		# build cross graph
		if self.dynamic_max_cross:
			cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
		else:
			cross_cutoff = self.cross_max_distance
		cross_edge_index, cross_edge_attr, cross_edge_sh, cross_edge_weight = self.build_cross_conv_graph(data, cross_cutoff)
		cross_lig, cross_rec = cross_edge_index
		cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)


		for l in range(len(self.lig_conv_layers)):
			# intra graph message passing
			lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
			lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh)

			# inter graph message passing
			rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
			lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
															  out_nodes=lig_node_attr.shape[0])

			if l != len(self.lig_conv_layers) - 1:
				rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
				rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh)

				lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
				rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_rec_edge_attr_,
																  cross_edge_sh, out_nodes=rec_node_attr.shape[0])

			# padding original features
			lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

			# update features with residual updates
			lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

			if l != len(self.lig_conv_layers) - 1:
				rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
				rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

		# compute translational and rotational score vectors
		center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
		center_edge_attr = self.center_edge_embedding(center_edge_attr)
		center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
		
		# with torch.cuda.amp.autocast(dtype=torch.float32):
		global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)
		
		tr_pred = global_pred[:, :3] + global_pred[:, 6:9]
		rot_pred = global_pred[:, 3:6] + global_pred[:, 9:]
		data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

		# fix the magnitude of translational and rotational score vectors
		tr_norm = clamped_norm(tr_pred, dim=1).unsqueeze(1)
		tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
		rot_norm = clamped_norm(rot_pred, dim=1).unsqueeze(1)
		rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))
		
		if self.scale_by_sigma:
			tr_pred = tr_pred / tr_sigma.unsqueeze(1)
			rot_pred = rot_pred * rot_sigma.unsqueeze(1)

		if self.no_torsion or data['ligand'].edge_mask.sum() == 0: 
			return tr_pred, rot_pred, torch.empty(0, device=tr_pred.device)

		# torsional components
		tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh, tor_edge_weight = self.build_bond_conv_graph(data)
		tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
		tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

		tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
		tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

		tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
								   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
		
		# with torch.cuda.amp.autocast(dtype=torch.float32):
		tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
							  out_nodes=data['ligand'].edge_mask.sum(), reduce='mean', edge_weight=tor_edge_weight)

		tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
		edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

		if self.scale_by_sigma:
			tor_pred = tor_pred * edge_sigma
		return tr_pred, rot_pred, tor_pred

	def build_lig_conv_graph(self, data):
		# builds the ligand graph edges and initial node and edge features
		data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['tr'])
		# (batch * N_node) x sigma_embed_dim (64)

		# compute edges
		radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)  # find edges with dis < 5
		edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
		edge_attr = torch.cat([
			data['ligand', 'ligand'].edge_attr,
			torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
		], 0)
		# edge_index: 2 * (bond + new[dis < 5])
		# edge_attr: (bond + new[dis < 5]) x 4

		# compute initial features
		edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]
		edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
		node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)

		src, dst = edge_index
		edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
		edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

		edge_attr = torch.cat([edge_attr, edge_length_emb], 1)      # # edge_attr: (bond + new[dis < 5]) x (4 + 64 + 64)
		edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
		edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)
		return node_attr, edge_index, edge_attr, edge_sh, edge_weight

	def build_rec_conv_graph(self, data):
		# builds the receptor initial node and edge embeddings
		data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['tr']) # tr rot and tor noise is all the same
		node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], 1)

		# this assumes the edges were already created in preprocessing since protein's structure is fixed
		edge_index = data['receptor', 'receptor'].edge_index
		src, dst = edge_index
		edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

		edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
		edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]
		edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
		edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
		edge_weight = self.get_edge_weight(edge_vec, self.rec_max_radius)
		return node_attr, edge_index, edge_attr, edge_sh, edge_weight

	def build_cross_conv_graph(self, data, cross_distance_cutoff):
		# builds the cross edges between ligand and receptor
		if torch.is_tensor(cross_distance_cutoff):
			# different cutoff for every graph (depends on the diffusion time)
			edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
								data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
								data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
		else:
			edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
							data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

		src, dst = edge_index
		edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

		edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
		edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
		edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
		edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
		cutoff_d = (
			cross_distance_cutoff[data["ligand"].batch[edge_index[0]]].squeeze()
			if torch.is_tensor(cross_distance_cutoff)
			else cross_distance_cutoff
		)
		edge_weight = self.get_edge_weight(edge_vec, cutoff_d)
		return edge_index, edge_attr, edge_sh, edge_weight

	def build_center_conv_graph(self, data):
		# builds the filter and edges for the convolution generating translational and rotational scores
		edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

		center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
		center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
		center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

		edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
		edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
		edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
		edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
		edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
		return edge_index, edge_attr, edge_sh

	def build_bond_conv_graph(self, data):
		# builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
		bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
		bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
		bond_batch = data['ligand'].batch[bonds[0]]
		edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)

		edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
		edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

		edge_attr = self.final_edge_embedding(edge_attr)
		edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
		edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)
		return bonds, edge_index, edge_attr, edge_sh, edge_weight

	def get_edge_weight(self, edge_vec, max_norm):
		if self.smooth_edges:
			normalised_norm = torch.clip(
				edge_vec.norm(dim=-1) * np.pi / max_norm, max=np.pi
			)
			return 0.5 * (torch.cos(normalised_norm) + 1.0).unsqueeze(-1)
		return 1.0
