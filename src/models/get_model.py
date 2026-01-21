# from .time_step_embedding import get_timestep_embedding
from .faster_TP_model import TensorProductModel
# from models import time_step_embedding

def get_vector_field(args):    
    lm_embedding_type = 'esm'
    tanh_out = getattr(args, 'tanh_out', False)
    print(tanh_out)
    model = TensorProductModel(
                        no_torsion=args.no_torsion,
                        num_conv_layers=args.num_conv_layers,
                        lig_max_radius=args.max_radius,
                        scale_by_sigma=args.scale_by_sigma,
                        sigma_embed_dim=args.sigma_embed_dim,
                        ns=args.ns, nv=args.nv,
                        distance_embed_dim=args.distance_embed_dim,
                        cross_distance_embed_dim=args.cross_distance_embed_dim,
                        dropout=args.dropout,
                        cross_max_distance=args.cross_max_distance,
                        dynamic_max_cross=args.dynamic_max_cross,
                        lm_embedding_type=lm_embedding_type, 
                        time_step_embedding_type=args.embedding_type,
                        time_step_embedding_scale=args.embedding_scale, 
                        tanh_out=tanh_out
                        )
    return model

