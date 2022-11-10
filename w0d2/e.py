import numpy as np
from fancy_einsum import einsum
from einops import reduce, rearrange, repeat
from typing import Union, Optional, Callable
import torch as t
import torchvision
import utils

arr = np.load("numbers.npy")

print(rearrange(arr, 'n c h w->c h (n w)').shape)

utils.display_array_as_img(rearrange(arr, 'n c h w->c h (n w)'))