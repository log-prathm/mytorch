import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.tensor import Tensor

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = p._zeros_like(p.data)

    def parameters(self):
        return []
        

class Neuron(Module):
    """
        Neuron(nonlin=False) → Linear neuron
        Neuron(nonlin=True) → Activated neuron
    """
    def __init__(self, in_dim, nonlin=False):
        self.w = Tensor(
            [random.uniform(-1, 1) for _ in range(in_dim)]
        )
        self.b = Tensor(0) # most framworks default to smaller or 0 bias. right now we'll use zero.
        self.nonlin = nonlin

    def __call__(self, x):

        # if x.shape != self.w.shape:
        #     raise ValueError(
        #         f"Expected {self.w.shape}, got {x.shape}"
        #     )

        out = (x * self.w).sum() + self.b
        if self.nonlin:
            return out.relu()
        else:
            return out
    
    def parameters(self):
        return [
            self.w,
            self.b
        ] 

class Perceptron(Module):
    pass

class Layer(Module):
    def __init__(self, in_dim, out_dim, nonlin=True):
        self.neurons = [Neuron(in_dim, nonlin=nonlin) for _ in range(out_dim)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]

        if len(out) == 1:
            return out[0]
        
        return out
    
    def parameters(self):
        params = []

        for n in self.neurons:
            params.extend(n.parameters())

        return (
            params
        )


class MLP(Module):

    def __init__(self, in_dim: int, out_dims: list[int]):
        sizes = [
            in_dim,
            *out_dims
        ]

        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],

                #don't apply for final layer
                nonlin=(i != len(out_dims)-1)
            ) for i in range(len(out_dims))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(
                layer.parameters()
            )

        return params
    
    
if "__main__" == __name__:
    pass