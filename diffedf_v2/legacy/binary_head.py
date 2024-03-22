from typing import List, Optional, Union, Tuple, Iterable, Callable, Dict
import math
import warnings
from tqdm import tqdm
from beartype import beartype

import torch
from e3nn import o3


from diffedf_v2.legacy import transforms
from diffedf_v2.legacy.equiformer.graph_attention_transformer import SeparableFCTP
from diffedf_v2.legacy.multiscale_tensor_field import MultiscaleTensorField
from diffedf_v2.legacy.gnn_data import FeaturedPoints, TransformPcd, set_featured_points_attribute, flatten_featured_points, detach_featured_points
from diffedf_v2.legacy.radial_func import SinusoidalPositionEmbeddings


class BinaryActionModelHead(torch.nn.Module):
    jittable: bool = True
    max_time: float
    time_emb_mlp: List[int]
    key_edf_dim: int
    query_edf_dim: int
    n_irreps_prescore: int
    edge_time_encoding: bool
    query_time_encoding: bool
    n_scales: int

    # @beartype
    def __init__(self,
                 n_binary_action: int,
                 n_pre_logit: int,
                 max_time: float,
                 time_emb_mlp: List[int],
                 key_tensor_field_kwargs: Dict,
                 irreps_query_edf: Union[str, o3.Irreps],
                 time_enc_n: float = 10000.,
                 edge_time_encoding: bool = False,
                 query_time_encoding: bool = True):
        super().__init__()
        if 'n_scales' in key_tensor_field_kwargs.keys():
            self.n_scales = key_tensor_field_kwargs['n_scales']
        else:
            self.n_scales = len(key_tensor_field_kwargs['r_cluster_multiscale'])

        ########### Time Encoder #############
        self.time_emb_mlp = time_emb_mlp
        self.irreps_time_emb = o3.Irreps(f"{self.time_emb_mlp[-1]}x0e")
        self.time_enc = SinusoidalPositionEmbeddings(dim=self.time_emb_mlp[0], max_val=max_time, n=time_enc_n)
        time_mlps_multiscale = torch.nn.ModuleList()
        for n in range(self.n_scales):
            time_mlp = []
            for i in range(1,len(time_emb_mlp)):
                time_mlp.append(torch.nn.Linear(self.time_emb_mlp[i-1], self.time_emb_mlp[i]))
                if i != len(time_emb_mlp) -1:
                    time_mlp.append(torch.nn.SiLU(inplace=True))
            time_mlp = torch.nn.Sequential(*time_mlp)
            time_mlps_multiscale.append(time_mlp)
        self.time_mlps_multiscale = time_mlps_multiscale
        if query_time_encoding:
            time_mlp = []
            for i in range(1,len(time_emb_mlp)):
                time_mlp.append(torch.nn.Linear(self.time_emb_mlp[i-1], self.time_emb_mlp[i]))
                if i != len(time_emb_mlp) -1:
                    time_mlp.append(torch.nn.SiLU(inplace=True))
            self.query_time_mlp = torch.nn.Sequential(*time_mlp)
        else:
            self.query_time_mlp = None
        self.time_emb_dim = time_emb_mlp[-1]

        self.edge_time_encoding = edge_time_encoding
        self.query_time_encoding = query_time_encoding
        if not self.edge_time_encoding and not self.query_time_encoding:
            raise NotImplementedError("No time encoding! Are you sure?")

        ################# Key field ########################
        if self.query_time_encoding:
            assert 'irreps_query' not in key_tensor_field_kwargs.keys()
            key_tensor_field_kwargs['irreps_query'] = str(self.irreps_time_emb)
        else:
            assert 'irreps_query' not in key_tensor_field_kwargs.keys()
            key_tensor_field_kwargs['irreps_query'] = None

        if self.edge_time_encoding:
            assert 'edge_context_emb_dim' not in key_tensor_field_kwargs.keys()
            key_tensor_field_kwargs['edge_context_emb_dim'] = self.time_emb_mlp[-1]
        else:
            assert 'edge_context_emb_dim' not in key_tensor_field_kwargs.keys()
            key_tensor_field_kwargs['edge_context_emb_dim'] = None

        self.key_tensor_field = MultiscaleTensorField(**key_tensor_field_kwargs)
        if self.query_time_encoding:
            assert self.key_tensor_field.use_dst_feature is True

        self.irreps_key_edf = self.key_tensor_field.irreps_output
        self.key_edf_dim = self.irreps_key_edf.dim

        if self.edge_time_encoding:
            assert self.time_emb_dim == self.key_tensor_field.context_emb_dim

        ##################### Query Transform ########################
        self.irreps_query_edf = o3.Irreps(irreps_query_edf)
        self.query_edf_dim = self.irreps_query_edf.dim
        self.query_transform = TransformPcd(irreps = self.irreps_query_edf)

        ##################### Tensor product for binary logits ###################
        self.n_pre_logit = n_pre_logit
        self.n_binary_action = n_binary_action
        self.logits_tp = SeparableFCTP(irreps_node_input = self.irreps_key_edf,
                                       irreps_edge_attr = self.irreps_query_edf,
                                       irreps_node_output = o3.Irreps(f"{n_pre_logit}x0e"),
                                       fc_neurons = None,
                                       use_activation = True,
                                       norm_layer = 'layer',
                                       # norm_layer = None,
                                       internal_weights = True)
        self.logits_mlp = torch.nn.Linear(n_pre_logit, n_binary_action)

    def forward(self, Ts: torch.Tensor,
                key_pcd_multiscale: List[FeaturedPoints],
                query_pcd: FeaturedPoints,
                time: torch.Tensor) ->  torch.Tensor:
        # assert torch.allclose(time, 0.) # dummy placeholder that is not removed due lazyness
        # time = time+0.5 # arbitrary value.

        # !!!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!
        # Batched forward is not yet implemented
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        assert Ts.ndim == 2 and Ts.shape[-1] == 7, f"{Ts.shape}" # Ts: (nT, 4+3: quaternion + position)
        assert time.ndim == 1 and len(time) == len(Ts), f"{time.shape}" # time: (nT,)
        assert query_pcd.f.ndim == 2 and query_pcd.f.shape[-1] == self.query_edf_dim, f"{query_pcd.f.shape}" # query_pcd: (nQ, 3), (nQ, F), (nQ,), (nQ)

        nT = len(Ts)
        nQ = len(query_pcd.x)

        query_weight = query_pcd.w     # (nQ,)
        assert isinstance(query_weight, torch.Tensor) # to tell torch.jit.script that it is tensor

        time_embs_multiscale: List[torch.Tensor] = []
        time_enc: torch.Tensor = self.time_enc(time)                       # (nT, time_emb_mlp[0])
        for time_mlp in self.time_mlps_multiscale:
            time_embs_multiscale.append(
                time_mlp(time_enc).unsqueeze(-2).expand(-1, nQ, -1).reshape(nT*nQ, self.time_emb_dim)        # (nT, time_emb_D) -> # (nT*nQ, time_emb_D)
            )

        ################# TODO: SCRUTINIZE THIS CODE ########################
        query_transformed: FeaturedPoints = self.query_transform(pcd = query_pcd, Ts = Ts)                                     # (nT, nQ, 3), (nT, nQ, F), (nT, nQ,), (nT, nQ,)
        query_features_transformed: torch.Tensor = query_transformed.f.clone()                                                 # (nT, nQ, F)
        if self.query_time_encoding:
            assert self.query_time_mlp is not None
            query_transformed = set_featured_points_attribute(points=query_transformed,
                                                                              f=self.query_time_mlp(time_enc).unsqueeze(-2).expand(nT, nQ, self.time_emb_dim),  # (nT, time_emb_D) -> # (nT, nQ, time_emb_D)
                                                                              w=None)    # (nT, nQ, 3), (nT, nQ, time_emb), (nT, nQ,), None
        else:
            query_transformed = set_featured_points_attribute(points=query_transformed, f=torch.empty_like(query_transformed.f), w=None)   # (nT, nQ, 3), (nT, nQ, -), (nT, nQ,), None

        query_transformed = flatten_featured_points(query_transformed)                                         # (nT*nQ, 3), (nT*nQ, time_emb), (nT*nQ,), None
        if self.edge_time_encoding:
            query_transformed = self.key_tensor_field(query_points = query_transformed,
                                                                      input_points_multiscale = key_pcd_multiscale,
                                                                      context_emb = time_embs_multiscale)                      # (nT*nQ, 3), (nT*nQ, F), (nT*nQ,), (nT*nQ,)
        else:
            assert self.query_time_encoding is True, f"You need to use at least one (query or edge) time encoding method."
            query_transformed = self.key_tensor_field(query_points = query_transformed,
                                                                      input_points_multiscale = key_pcd_multiscale,
                                                                      context_emb = None)                                         # (nT*nQ, 3), (nT*nQ, F), (nT*nQ,), (nT*nQ,)                                                         # (nT*nQ, F)
        key_features: torch.Tensor = query_transformed.f
        query_features_transformed = query_features_transformed.view(-1, query_features_transformed.shape[-1])                    # (nT*nQ, F)

        ######################################################################

        pre_logits: torch.Tensor = self.logits_tp(query_features_transformed, key_features,   # (nT*nQ, n_pre_dim)
                                                  edge_scalars = None, batch=None,)
        pre_logits = pre_logits.view(nT, nQ, self.n_pre_logit)                                # (nT, nQ, n_action_dim)
        logits: torch.Tensor = self.logits_mlp(pre_logits)                                    # (nT, nQ, n_action_dim)
        logits = torch.einsum('q,tqa->ta', query_weight, logits) # (nT, n_action_dim)

        return logits # (nT, n_action_dim)

    # @torch.jit.export
    def warmup(self, Ts: torch.Tensor,
               key_pcd_multiscale: List[FeaturedPoints],
               query_pcd: FeaturedPoints,
               time: torch.Tensor) -> torch.Tensor:
        return self.forward(Ts=Ts, key_pcd_multiscale=key_pcd_multiscale, query_pcd=query_pcd, time=time)

    # @torch.jit.ignore
    def _get_fake_input(self):
        device = next(iter(self.parameters())).device

        from diffedf_v2.legacy.transforms import random_quaternions
        nT = 5
        nP = 100
        nQ = 10
        Ts = torch.cat([random_quaternions(nT, device=device), torch.randn(nT, 3, device=device)], dim=-1)
        time= torch.rand(nT, device=device)


        key_pcd_multiscale = [
            FeaturedPoints(
                x=torch.randn(nP,3, device=device, ),
                f=o3.Irreps(self.irreps_key_edf).randn(nP,-1, device=device, ),
                b=torch.zeros(nP, device=device, dtype=torch.long)
            ) for _ in range(self.n_scales)
        ]
        query_pcd= FeaturedPoints(
                x=torch.randn(nQ,3, device=device, ),
                f=o3.Irreps(self.irreps_key_edf).randn(nQ,-1, device=device, ),
                b=torch.zeros(nQ, device=device, dtype=torch.long),
                w=torch.ones(nQ, device=device, )
            )

        return Ts, key_pcd_multiscale, query_pcd, time