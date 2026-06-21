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
        Neuron: output = weighted sum + bias
        Perceptron: y = f(x*w + b) <-- activation
    """
    def __init__(self, in_dim):
        self.w = Tensor(
            [random.uniform(-1, 1) for _ in range(in_dim)]
        )
        self.b = Tensor(0) # most framworks default to smaller or 0 bias. right now we'll use zero.

    def __call__(self, x):

        return (x * self.w).sum() + self.b
    
    def parameters(self):
        return [
            self.w,
            self.b
        ] 

class Perceptron(Module):
    pass

class Linear(Module):
    pass

class MLP(Module):
    pass
    
    
if "__main__" == __name__:
    pass