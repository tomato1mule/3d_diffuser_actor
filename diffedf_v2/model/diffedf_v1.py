from typing import List, Optional, Union, Tuple, Iterable, Callable, Dict, Sequence
import math
import warnings
from tqdm import tqdm
from beartype import beartype

import torch
from e3nn import o3


from diffedf_v2.model import BaseDiffusionEDF
from diffedf_v2.legacy.score_head import ScoreModelHead
from diffedf_v2.legacy.binary_head import BinaryActionModelHead


class DiffusionEDFv1(BaseDiffusionEDF):

    def init_sbm(self): # sbm is acronym for [S]core-[b]ased [M]odel
        self._match_sbm_cfg()
        self.sbm = ScoreModelHead(**self.sbm_cfg)

    def _match_sbm_cfg(self):
        if 'irreps_query_edf' in self.sbm_cfg.keys():
            assert self.sbm_cfg['irreps_query_edf'] == self.tool_enc.irreps_output, f"{self.sbm_cfg['irreps_query_edf']} != {self.tool_enc.irreps_output}"
        else:
            self.sbm_cfg['irreps_query_edf'] = self.tool_enc.irreps_output

        if 'irreps_input' in self.sbm_cfg['key_tensor_field_kwargs'].keys():
            assert self.sbm_cfg['key_tensor_field_kwargs']['irreps_input'] == self.scene_enc.irreps_output, f"{self.sbm_cfg['key_tensor_field_kwargs']} != {self.scene_enc.irreps_output}"
        else:
            self.sbm_cfg['key_tensor_field_kwargs']['irreps_input'] = self.scene_enc.irreps_output

        # Disable deprecated settings.
        assert 'use_src_point_attn' not in self.sbm_cfg['key_tensor_field_kwargs'].keys()
        self.sbm_cfg['key_tensor_field_kwargs']['use_src_point_attn'] = False
        assert 'use_dst_point_attn' not in self.sbm_cfg['key_tensor_field_kwargs'].keys()
        self.sbm_cfg['key_tensor_field_kwargs']['use_dst_point_attn'] = False

    def init_bin_act_head(self):
        if self.bin_act_cfg:
            self._match_bin_act_cfg()
            self.bin_act_head = BinaryActionModelHead(**self.bin_act_cfg)
        else:
            self.bin_act_head = None

    def _match_bin_act_cfg(self):
        if 'irreps_query_edf' in self.bin_act_cfg.keys():
            assert self.bin_act_cfg[
                       'irreps_query_edf'] == self.tool_enc.irreps_output, f"{self.bin_act_cfg['irreps_query_edf']} != {self.tool_enc.irreps_output}"
        else:
            self.bin_act_cfg['irreps_query_edf'] = self.tool_enc.irreps_output

        if 'irreps_input' in self.bin_act_cfg['key_tensor_field_kwargs'].keys():
            assert self.bin_act_cfg['key_tensor_field_kwargs'][
                       'irreps_input'] == self.scene_enc.irreps_output, f"{self.bin_act_cfg['key_tensor_field_kwargs']} != {self.scene_enc.irreps_output}"
        else:
            self.bin_act_cfg['key_tensor_field_kwargs']['irreps_input'] = self.scene_enc.irreps_output

        # Disable deprecated settings.
        assert 'use_src_point_attn' not in self.bin_act_cfg['key_tensor_field_kwargs'].keys()
        self.bin_act_cfg['key_tensor_field_kwargs']['use_src_point_attn'] = False
        assert 'use_dst_point_attn' not in self.bin_act_cfg['key_tensor_field_kwargs'].keys()
        self.bin_act_cfg['key_tensor_field_kwargs']['use_dst_point_attn'] = False