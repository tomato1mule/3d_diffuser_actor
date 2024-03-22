from functools import partial
from typing import get_type_hints, Optional, List, Dict, Tuple, Union, Any
from copy import deepcopy

from prettytable import PrettyTable

import torch
from torch import Tensor

Device = Union[torch.device, str]

class GeometryBase():
    _data_registry: List[str] = ['name']
    _device: Device
    
    def __init__(self):
        if __debug__:
            for dat in self.get_data_registry():
                assert dat in get_type_hints(self.__class__).keys()
    
    @property
    def device(self) -> Device:
        return self._device
    
    @classmethod
    def get_data_registry(cls):
        return deepcopy(cls._data_registry)
    
    def to(self, *args, **kwargs):
        for attr in self.get_data_registry():
            data = self.__getattribute__(attr)
            if hasattr(data, 'to'):
                self.__setattr__(attr, data.to(*args, **kwargs))
        return self
    
    def _print_data(self, keys, shape_only=True, print_none=False) -> str:       
        data_name = ""
        
        table = PrettyTable(['Key', 'Shape/Type'] if shape_only else ['Key', 'Shape/Type', 'Value'])
        for key in keys:
            val = self.__getattribute__(key)
            if key == 'name':
                data_name = val
                continue
            
            if hasattr(val, 'shape'):
                shape = tuple(val.shape)
            elif val is None:
                if print_none:
                    shape = 'None'
                    val = '---'
                else:
                    continue
            else:
                shape = type(val).__name__
                
            if shape_only:
                table.add_row([key, shape])
            else:
                table.add_row([key, shape, val])        
        
        repr: str = f"{self.__class__.__name__}"
        if data_name:
            repr += f" (name: {data_name})"
        repr += "\n"
        repr += table.__str__()
        return repr
    
    def __repr__(self, shape_only=True, print_none=False) -> str:
        keys = self.get_data_registry()
        return self._print_data(keys, shape_only=shape_only, print_none=print_none)

    def __str__(self, *args, **kwargs) -> str:
        return self.__repr__(*args, **kwargs)

    def printval(self, **kwargs) -> str:
        return print(self.__repr__(shape_only=False, **kwargs))
    
    def jsonify(self, exclude_empty = True) -> Dict[str, Any]:
        state = {}
        for key in self.get_data_registry():
            val = self.__getattribute__(key)
            
            if hasattr(val, 'tolist'):
                state[key] = val.tolist()
            elif hasattr(val, 'jsonify'):
                state[key] = val.jsonify()
            elif not val and exclude_empty:
                continue
            else:
                state[key] = val
        return state
    
    

class GraphEdge(GeometryBase):
    _data_registry: List[str] = ['edge_src', 'edge_dst', 'edge_disp', 'edge_length', 'edge_sh', 'edge_feature', 'edge_scalars', 'edge_weights', 'name']
    edge_src: Optional[Tensor] = None
    edge_dst: Optional[Tensor] = None
    edge_disp: Optional[Tensor] = None
    edge_length: Optional[Tensor] = None
    edge_sh: Optional[Tensor] = None
    edge_feature: Optional[Tensor] = None
    edge_scalars: Optional[Tensor] = None
    edge_weights: Optional[Tensor] = None
    name: str = ""
    
    def __init__(self, 
                 edge_src: Optional[Tensor] = None, 
                 edge_dst: Optional[Tensor] = None, 
                 edge_disp: Optional[Tensor] = None, 
                 edge_length: Optional[Tensor] = None, 
                 edge_sh: Optional[Tensor] = None, 
                 edge_feature: Optional[Tensor] = None, 
                 edge_scalars: Optional[Tensor] = None, 
                 edge_weights: Optional[Tensor] = None,
                 name: str = ""):
        super().__init__()
        self.edge_src=edge_src
        self.edge_dst=edge_dst
        self.edge_disp=edge_disp
        self.edge_length=edge_length
        self.edge_sh=edge_sh
        self.edge_feature=edge_feature
        self.edge_scalars=edge_scalars
        self.edge_weights=edge_weights
        self.name=name

class PointGraph(GeometryBase):
    _data_registry: List[str] = ['x', 'b', 'f', 'w', 'edge', 'name']
    x: Optional[Tensor] # node position
    b: Optional[Tensor] # node batch
    f: Optional[Tensor] = None # node feature
    w: Optional[Tensor] = None # node weight
    edge: Optional[GraphEdge] = None # internal edge
    name: str = ""
    
    def __init__(self,
                 x: Optional[Tensor] = None, 
                 b: Optional[Tensor] = None, 
                 f: Optional[Tensor] = None, 
                 w: Optional[Tensor] = None,
                 name: str = ""):
        super().__init__()
        self.x=x
        self.b=b
        self.f=f
        self.w=w
        self.name=name
    
class HeteroPointGraph():
    graphs: List[Optional[PointGraph]]
    edges: Dict[Tuple[PointGraph, PointGraph], GraphEdge]
    name: str = ""
    
    def __init__(self, 
                 graphs: Optional[List[Optional[PointGraph]]] = None, 
                 edges: Optional[List[Optional[PointGraph]]] = None, 
                 name: str = ""):
        super().__init__()
        if graphs is None:
            graphs = []
        if edges is None:
            edges = {}
        
        self.graphs=graphs
        self.edges=edges
        self.name=name
        
    def to(self, *args, **kwargs):
        raise NotImplementedError
    
    
    