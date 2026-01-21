import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
import numpy as np
from e3nn.nn import BatchNorm, Activation


def clamped_norm(vec, dim=1, min=1e-6):
    return torch.clamp(torch.linalg.vector_norm(vec, dim=dim), min=min)

class Output_scale(nn.Module):
	def __init__(self, scale=math.pi):
		super(Output_scale, self).__init__()
		self.scale = scale
		self.Tanh = nn.Tanh()

	def forward(self, x):
		return self.Tanh(x) * self.scale

class AtomEncoder(torch.nn.Module):
	def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type= None):
		# first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
		super(AtomEncoder, self).__init__()
		self.atom_embedding_list = torch.nn.ModuleList()
		self.num_categorical_features = len(feature_dims[0])
		self.num_scalar_features = feature_dims[1] + sigma_embed_dim
		self.lm_embedding_type = lm_embedding_type
		for i, dim in enumerate(feature_dims[0]):
			emb = torch.nn.Embedding(dim, emb_dim)
			torch.nn.init.xavier_uniform_(emb.weight.data)
			self.atom_embedding_list.append(emb)

		if self.num_scalar_features > 0:
			self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
		if self.lm_embedding_type is not None:
			if self.lm_embedding_type == 'esm':
				self.lm_embedding_dim = 1280
			else: 
				raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)
			self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)

	def forward(self, x):
		x_embedding = 0
		if self.lm_embedding_type is not None:
			assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
		else:
			assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
		for i in range(self.num_categorical_features):
			x_embedding += self.atom_embedding_list[i](x[:, i].long())

		if self.num_scalar_features > 0:
			x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
		if self.lm_embedding_type is not None:
			x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim:]], axis=1))
		return x_embedding


class zero_one_BatchNorm(BatchNorm):
	def forward(self, input):
		self.running_mean = torch.zeros_like(self.running_mean)
		self.running_var = torch.ones_like(self.running_var)
		output = super().forward(input)
		return output

class OldTensorProductConvLayer(torch.nn.Module):
	def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
				 hidden_features=None, zero_one_bn=False):
		super(OldTensorProductConvLayer, self).__init__()
		self.in_irreps = in_irreps
		self.out_irreps = out_irreps
		self.sh_irreps = sh_irreps
		self.residual = residual
		if hidden_features is None:
			hidden_features = n_edge_features

		self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

		self.fc = nn.Sequential(
			nn.Linear(n_edge_features, hidden_features),
			nn.Mish(),
			nn.Dropout(dropout),
			nn.Linear(hidden_features, tp.weight_numel)
		)
		bn_func = zero_one_BatchNorm if zero_one_bn else BatchNorm
		self.batch_norm = bn_func(out_irreps, eps=1e-4) if batch_norm else None

	def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean', sfmx=False):
		edge_src, edge_dst = edge_index
		# edge_attrs = 
		# ix = 0
		# for weight in self.tp.weight_views(edge_attrs):
		# 	bs, row, col, num = weight.shape
		# 	numel = row * col * num
		# 	edge_attrs[:, ix: ix + numel] = edge_attrs[:, ix: ix + numel].view(bs, row * col, num).softmax(1).view(bs, numel)
		# 	ix += numel

		tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

		out_nodes = out_nodes or node_attr.shape[0]
		out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

		if self.residual:
			padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
			out = out + padded
			
		if sfmx:
			N, ns2 = out.shape
			assert ns2 % 2 == 0
			out = out.view(N, 2, ns2 // 2).softmax(2).view(N, ns2)

		if self.batch_norm:
			out = self.batch_norm(out.float())
		return out

class GaussianSmearing(torch.nn.Module):
	# used to embed the edge distances
	def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
		super().__init__()
		offset = torch.linspace(start, stop, num_gaussians)
		self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
		self.register_buffer('offset', offset)

	def forward(self, dist):
		dist = dist.view(-1, 1) - self.offset.view(1, -1)
		return torch.exp(self.coeff * torch.pow(dist, 2))