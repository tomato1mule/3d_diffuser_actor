from typing import List, Optional, Union, Tuple, Iterable, Callable, Dict
import math
import warnings
from tqdm import tqdm
from beartype import beartype

import torch
from e3nn import o3


from diffedf_v2.legacy import transforms
from diffedf_v2.legacy.equiformer.graph_attention_transformer import SeparableFCTP
from diffedf_v2.legacy.unet_feature_extractor import UnetFeatureExtractor
from diffedf_v2.legacy.multiscale_tensor_field import MultiscaleTensorField
from diffedf_v2.legacy.keypoint_extractor import KeypointExtractor, StaticKeypointModel
from diffedf_v2.legacy.gnn_data import FeaturedPoints, TransformPcd, set_featured_points_attribute, flatten_featured_points, detach_featured_points
from diffedf_v2.legacy.radial_func import SinusoidalPositionEmbeddings
from diffedf_v2.legacy.score_head import ScoreModelHead
from diffedf_v2.legacy.score_model_base import ScoreModelBase
from diffedf_v2.legacy.score_head_ebm import EbmScoreModelHead


class MultiscaleScoreModel(ScoreModelBase):

    @beartype
    def __init__(self, 
                 query_model: str,
                 score_head_kwargs: Dict,
                 key_kwargs: Dict,
                 query_kwargs: Dict,
                 deterministic: bool = False):
        super().__init__()
        key_feature_extractor_kwargs = key_kwargs['feature_extractor_kwargs']
        key_feature_extractor_name = key_kwargs['feature_extractor_name']

        print("ScoreModel: Initializing Key Feature Extractor")
        if key_feature_extractor_name == 'UnetFeatureExtractor':
            self.key_model = UnetFeatureExtractor(
                **(key_feature_extractor_kwargs),
                deterministic=deterministic
            )
        elif key_feature_extractor_name == 'ForwardOnlyFeatureExtractor':
            # from diffedf_v2.diffedf_v1.forward_only_feature_extractor import ForwardOnlyFeatureExtractor
            # self.key_model = ForwardOnlyFeatureExtractor(
            #     **(key_feature_extractor_kwargs),
            #     deterministic=deterministic
            # )
            raise NotImplementedError(f"'ForwardOnlyFeatureExtractor' is Deprecated")
        else:
            raise ValueError(f"Unknown feature extractor name: {key_feature_extractor_name}")
        
        print("ScoreModel: Initializing Query Model")
        if query_model == 'KeypointExtractor':
            self.query_model = KeypointExtractor(
                **(query_kwargs),
                deterministic=deterministic
            )
        elif query_model == 'StaticKeypointModel':
            self.query_model = StaticKeypointModel(
                **(query_kwargs),
            )
        else:
            raise ValueError(f"Unknown query model: {query_model}")
        
        max_time: float = float(score_head_kwargs['max_time'])
        time_emb_mlp: List[int] = score_head_kwargs['time_emb_mlp']
        if 'lin_mult' in score_head_kwargs.keys():
            lin_mult: float = float(score_head_kwargs['lin_mult'])
        else:
            raise NotImplementedError()
            lin_mult: float = float(1.)
        if 'ang_mult' in score_head_kwargs.keys():
            ang_mult: float = float(score_head_kwargs['ang_mult'])
        else:
            raise NotImplementedError()
            ang_mult: float = math.sqrt(2.)
        edge_time_encoding: bool = score_head_kwargs['edge_time_encoding']
        query_time_encoding: bool = score_head_kwargs['query_time_encoding']

        key_tensor_field_kwargs = score_head_kwargs['key_tensor_field_kwargs']
        assert 'irreps_input' not in key_tensor_field_kwargs.keys()
        key_tensor_field_kwargs['irreps_input'] = self.key_model.irreps_output
        assert 'use_src_point_attn' not in key_tensor_field_kwargs.keys()
        key_tensor_field_kwargs['use_src_point_attn'] = False
        assert 'use_dst_point_attn' not in key_tensor_field_kwargs.keys()
        key_tensor_field_kwargs['use_dst_point_attn'] = False

        use_ebm_score_head: bool = score_head_kwargs.get("ebm", False)
        if use_ebm_score_head:
            print("EbmScoreModel: Initializing Score Head")
            self.score_head = EbmScoreModelHead(max_time=max_time, 
                                                time_emb_mlp=time_emb_mlp,
                                                key_tensor_field_kwargs=key_tensor_field_kwargs,
                                                irreps_query_edf=self.query_model.irreps_output,
                                                lin_mult=lin_mult,
                                                ang_mult=ang_mult,
                                                edge_time_encoding=edge_time_encoding,
                                                query_time_encoding=query_time_encoding,
                                                )
        else:
            print("ScoreModel: Initializing Score Head")
            self.score_head = ScoreModelHead(max_time=max_time, 
                                            time_emb_mlp=time_emb_mlp,
                                            key_tensor_field_kwargs=key_tensor_field_kwargs,
                                            irreps_query_edf=self.query_model.irreps_output,
                                            lin_mult=lin_mult,
                                            ang_mult=ang_mult,
                                            edge_time_encoding=edge_time_encoding,
                                            query_time_encoding=query_time_encoding,
                                            )

        self.lin_mult = self.score_head.lin_mult
        self.ang_mult = self.score_head.ang_mult

    def get_key_pcd_multiscale(self, pcd: FeaturedPoints) -> List[FeaturedPoints]:
        return self.key_model(pcd)
    
    def get_query_pcd(self, pcd: FeaturedPoints) -> FeaturedPoints:
        return self.query_model(pcd)