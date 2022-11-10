import torch as t
import utils

def as_strided_trace(mat: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    n = mat.clone()
    tr_n = t.as_strided(n, (mat.shape[0],), (mat.shape[0] + 1,)).sum()
    return tr_n

utils.test_trace(as_strided_trace)

# def as_strided_mv(mat: t.Tensor, vec: t.Tensor) -> t.Tensor:
#     '''
#     Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
#     '''
#     return (t.as_strided(mat, (16, 16, 16), (16, 1, 0)) * t.as_strided(b, (16, 16, 16), (0, 16, 1))).sum(1)


# utils.test_mv(as_strided_mv)
# utils.test_mv2(as_strided_mv)

def as_strided_mm(matA: t.Tensor, matB: t.Tensor) -> t.Tensor:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
        # Get the matrix strides, and matrix dims
    sAs = list(matA.stride())
    dAs = matA.shape
    sBs = list(matB.stride())
    dBs = matB.shape
    
    expanded_size = dAs + dBs[1:] 
    
    matA_expanded_stride = sAs + [0]
    matA_expanded = matA.as_strided(expanded_size, matA_expanded_stride)
    
    matB_expanded_stride = [0] + sBs
    matB_expanded = matB.as_strided(expanded_size, matB_expanded_stride)
    
    product_expanded = matA_expanded * matB_expanded
    return product_expanded.sum(dim=1)


utils.test_mm(as_strided_mm)
utils.test_mm2(as_strided_mm)