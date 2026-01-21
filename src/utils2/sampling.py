import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..datasets.data_utils import modify_conformer, set_time
from .torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R

from ..datasets.process_mols import write_mol_with_coords

def randomize_position(data_list, no_torsion, no_random, tr_sigma_max):
    # in place modification of the list
    if not no_torsion:
        # randomize torsion angles
        for complex_graph in data_list:
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
            complex_graph['ligand'].pos = \
                modify_conformer_torsion_angles(complex_graph['ligand'].pos,
                                                complex_graph['ligand', 'ligand'].edge_index.T[
                                                    complex_graph['ligand'].edge_mask],
                                                complex_graph['ligand'].mask_rotate[0], torsion_updates)

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float()
        
        # Add prior position to the ligand
        complex_graph['ligand'].pos = (complex_graph['ligand'].pos - molecule_center) @ random_rotation.T
        
        # 
        p2rank_prior = complex_graph.ligand_center.view(-1, 3)
        idx = torch.randint(0, p2rank_prior.size(0), size=()).item()
        prior = p2rank_prior[idx].view(1, 3)
        complex_graph['ligand'].pos += prior
        if not no_random:  # note for now the torsion angles are still randomised  # if False:
            if prior.abs().sum() < 1e-5:
                tr_sigma_max = 19
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))     # 
            complex_graph['ligand'].pos += tr_update

        # Cut protein
        # Dist between rec and lig
        # ligand_pos = complex_graph['ligand'].pos
        receptor_pos = complex_graph['receptor'].pos
        # diff = receptor_pos.unsqueeze(1) - ligand_pos.unsqueeze(0) # N_rec x N_lig x 3
        # dist = torch.linalg.norm(diff, dim=-1)
        # min_dist = dist.min(dim=1).values

        # Dist between rec and prior
        diff_prior = receptor_pos.unsqueeze(1) - prior.unsqueeze(0)
        dist_prior = torch.linalg.norm(diff_prior, dim=-1)
        min_dist_prior = dist_prior.min(dim=1).values

        # filter
        cutoff_lig = 20
        receptor_mask = min_dist_prior < (cutoff_lig)
        kept_idx = receptor_mask.nonzero(as_tuple=False).view(-1)

        if kept_idx.numel() == 0:
            print(f"Skip one complex due to empty receptor region for {complex_graph.name[0]}, original {receptor_pos.shape[0]}")
            continue
        # update receptor 
        complex_graph['receptor'].x = complex_graph['receptor'].x[kept_idx]
        complex_graph['receptor'].pos = receptor_pos[kept_idx]
        if 'mu_r_norm' in complex_graph['receptor']:
            complex_graph['receptor'].mu_r_norm = complex_graph['receptor'].mu_r_norm[kept_idx]
        if 'side_chain_vecs' in complex_graph['receptor']:
            complex_graph['receptor'].side_chain_vecs = complex_graph['receptor'].side_chain_vecs[kept_idx]

        # rebuild edge
        edge_index = complex_graph['receptor', 'rec_contact', 'receptor'].edge_index
        edge_mask = receptor_mask[edge_index[0]] & receptor_mask[edge_index[1]]
        edge_index = edge_index[:, edge_mask]
        
        old_to_new = -torch.ones(receptor_mask.shape[0], dtype=torch.long)
        old_to_new[kept_idx] = torch.arange(kept_idx.shape[0])
        new_edge_index = old_to_new[edge_index]

        complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = new_edge_index

        # update batch, ptr
        complex_graph['receptor'].batch = complex_graph['receptor'].batch[kept_idx]
        complex_graph['receptor'].ptr = torch.tensor([0, kept_idx.shape[0]], dtype=torch.long)


@torch.no_grad()
def sampling_fm(data_list, model, inference_steps, tr_schedule, rot_schedule, tor_schedule, device, model_args,
             visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32):
    N = len(data_list)
    for t_idx in range(inference_steps):
        t_tr, t_rot, t_tor = tr_schedule[t_idx], rot_schedule[t_idx], tor_schedule[t_idx]
        dt_tr = tr_schedule[t_idx + 1] - tr_schedule[t_idx] if t_idx < inference_steps - 1 else 1- tr_schedule[t_idx]
        dt_rot = rot_schedule[t_idx + 1] - rot_schedule[t_idx] if t_idx < inference_steps - 1 else 1- rot_schedule[t_idx]
        dt_tor = tor_schedule[t_idx + 1] - tor_schedule[t_idx] if t_idx < inference_steps - 1 else 1- tor_schedule[t_idx]

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            set_time(complex_graph_batch, t_tr, t_rot, t_tor, b, model_args.all_atoms, device)

            vt_tr, vt_rot, vt_tor = model(complex_graph_batch)

            # translation
            tr_perturb = (vt_tr * dt_tr).cpu()

            # rotation
            rot_perturb = (vt_rot * dt_rot).cpu()
            # print('rot_perturb', rot_perturb, 'rot_perturb.norm', rot_perturb.norm(dim=1))
            # torsion
            if not model_args.no_torsion:
                tor_perturb = (vt_tor * dt_tor).cpu().numpy()
                # print('tor_perturb', tor_perturb)
                torsions_per_molecule = tor_perturb.shape[0] // b
            else:
                tor_perturb = None

            # Apply noise
            new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1], rot_perturb[i:i + 1].squeeze(0),
                                          tor_perturb[i * torsions_per_molecule:(i + 1) * torsions_per_molecule] if not model_args.no_torsion else None) # type: ignore
                         for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
            data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    if confidence_model is not None:
        loader = DataLoader(data_list, batch_size=batch_size)
        confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size)) # type: ignore
        confidence = []
        for complex_graph_batch in loader:
            complex_graph_batch = complex_graph_batch.to(device)
            if confidence_data_list is not None:
                confidence_complex_graph_batch = next(confidence_loader).to(device)
                confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                set_time(confidence_complex_graph_batch, 0, 0, 0, N, confidence_model_args.all_atoms, device) # type: ignore
                confidence.append(confidence_model(confidence_complex_graph_batch))
            else:
                confidence.append(confidence_model(complex_graph_batch))
        confidence = torch.cat(confidence, dim=0)
    else:
        confidence = None
    return data_list, confidence