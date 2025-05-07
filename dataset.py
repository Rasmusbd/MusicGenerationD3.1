import torch
import numpy as np
from typing import Union, Optional
from deeptrack.image import Image

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 pipeline,
                 inputs=None,
                 length=None,
                 replace: Union[bool, float] = False,
                 float_dtype: Optional[Union[torch.dtype, str]] = "default"):
        self.pipeline = pipeline
        self.replace = replace

        if inputs is None:
            if length is None:
                raise ValueError("Either inputs or length must be specified.")
            else:
                inputs = [[]] * length
        self.inputs = inputs
        self._cache = [None] * len(self.inputs)

        if float_dtype == "default":
            float_dtype = torch.get_default_dtype()
        self.float_dtype = float_dtype

    def __getitem__(self, index):
        self.pipeline.update()
        res = self.pipeline(self.inputs[index])

        if not isinstance(res, (tuple, list)):
            res = (res,)
        
        res = tuple(r._value if isinstance(r, Image) else r for r in res)
        res = tuple(self._as_tensor(r) for r in res)
        return res

    def _as_tensor(self, x):
        if isinstance(x, (int, float, bool)):
            x = torch.from_numpy(np.array([x]))
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.ndim > 2 and x.dtype not in [np.uint8, np.uint16, np.uint32, np.uint64]:
                x = x.permute(-1, *range(x.ndim - 1))
        if isinstance(x, Image):
            return self._as_tensor(x._value)
        else:
            x = torch.Tensor(x)

        if self.float_dtype and x.dtype in [torch.float16, torch.float32, torch.float64]:
            x = x.to(self.float_dtype)
        if x.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            x = x.to(torch.long)

        return x
    """
    def _should_replace(self, index):
        if isinstance(self.replace, bool):
            return self.replace
        elif callable(self.replace):
            try:
                return self.replace()
            except TypeError:
                return self.replace(index)
        elif isinstance(self.replace, float) and 0 <= self.replace <= 1:
            return np.random.rand() < self.replace
        else:
            return False
    """
    def __len__(self):
        return len(self.inputs)
