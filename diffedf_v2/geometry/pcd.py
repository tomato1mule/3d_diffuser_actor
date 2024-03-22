from functools import partial
from typing import Optional

from prettytable import PrettyTable

import torch
from torch import Tensor



class Points():
    x: Tensor # Position
    b: Tensor # Batch idx
    w: Optional[Tensor] = None # (Optional) weight
    N: int    # Number of points
    B: int    # Number of batches
    
    def __init__(self, x: Tensor, b: Optional[Tensor]=None, w: Optional[Tensor]=None):
        """_summary_

        Args:
            x (Tensor): position of 3D points, size (N,3).
            b (Tensor, optional): batch idx of points, size (N). If not specified, every points are in zeroth batch.
            w (Tensor, optional): optional weight of points.
        """
        
        if __debug__:
            assert isinstance(x, Tensor)
            assert len(x.shape) == 2 and x.shape[-1]==3, f"{x.shape}"
            if b is not None:
                assert isinstance(b, Tensor), f"{type(b)}"
                assert len(b.shape) == 1 and b.shape[0]==x.shape[0], f"{b.shape} || {x.shape}"
            if w is not None:
                assert isinstance(w, Tensor), f"{type(w)}"
                assert len(w.shape) == 1 and w.shape[0]==x.shape[0], f"{w.shape} || {x.shape}"
        
        self.x = x
        if b is None:
            self.b = torch.zeros_like(x[...,0], dtype=torch.long)
        else:
            self.b = b
        self.w = w
        
    @property
    def N(self) -> int:
        return self.x.shape[0]
    
    @property
    def B(self) -> int:
        return torch.max(self.b).item()+1
    
    @property
    def hasWeight(self) -> bool:
        if self.w is None:
            return False
        else:
            return True
        
    def new(self, **kwargs):
        _kwargs = {'x': self.x, 'b': self.b, 'w': self.w}
        for k,v in kwargs.items():
            _kwargs[k]=v
        return Points(**_kwargs)
    
    def _print(self, keys, shape_only=True) -> str:
        table = PrettyTable(['Key', 'Shape'] if shape_only else ['Key', 'Shape', 'Value'])
        for key in keys:
            val = self.__getattribute__(key)
            if shape_only:
                table.add_row([key, tuple(val.shape) if hasattr(val, 'shape') else '---'])
            else:
                table.add_row([key, tuple(val.shape) if hasattr(val, 'shape') else '---', val])        
        
        repr: str = f"{self.__class__.__name__}"
        repr += "\n"
        repr += table.__str__()
        return repr
    
    def __repr__(self, shape_only=True) -> str:
        keys = ['x', 'w', 'b']
        return self._print(keys, shape_only=shape_only)

    def __str__(self, *args, **kwargs) -> str:
        return self.__repr__(*args, **kwargs)

    def printval(self) -> str:
        return print(self.__repr__(shape_only=False))


class FeaturedPoints(Points):
    x: Tensor # Position
    f: Tensor # Feature
    b: Tensor # Batch idx
    w: Optional[Tensor] = None # (Optional) weight
    N: int    # Number of points
    F: int    # Feature dimension
    B: int    # Number of batches
    
    def __init__(self, x: Tensor, f: Tensor, b: Optional[Tensor]=None, w: Optional[Tensor]=None):
        """_summary_

        Args:
            x (Tensor): position of 3D points, size (N,3).
            f (Tensor): position of 3D points, size (N,dimF).
            b (Tensor, optional): batch idx of points, size (N). If not specified, every points are in zeroth batch.
            w (Tensor, optional): optional weight of points.
        """
        
        super().__init__(x=x, b=b, w=w)
        
        if __debug__:
            assert isinstance(f, Tensor)
            assert len(f.shape) == 2 and f.shape[0]==x.shape[0], f"{f.shape} || {x.shape}"
        self.f = f
   
    @property
    def F(self) -> int:
        return self.f.shape[-1]
        
    def new(self, **kwargs):
        _kwargs = {'x': self.x, 'f': self.f, 'b': self.b, 'w': self.w}
        for k,v in kwargs.items():
            _kwargs[k]=v
        return FeaturedPoints(**_kwargs)
    
    def __repr__(self, shape_only=True) -> str:
        keys = ['x', 'f', 'w', 'b']
        return self._print(keys, shape_only=shape_only)