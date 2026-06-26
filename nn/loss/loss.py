from core.tensor import Tensor

'''
    Loss is not a new kind of Tensor
    They are just functions that return Tensor
'''

def mse(preds: Tensor, target: Tensor) -> Tensor:

    return ((preds - target) ** 2).sum()

# def mae(preds: Tensor, target: Tensor) -> Tensor:

#     return (preds - target).abs().mean()