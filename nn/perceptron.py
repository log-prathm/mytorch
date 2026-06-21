import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.tensor import Tensor

class Perceptron:

    def __init__(self, in_dim):
        self.w = Tensor(
            [random.uniform(-1, 1) for _ in range(in_dim)]
        )
        self.b = Tensor(0) # most framworks default to smaller or 0 bias. right now we'll use zero.

    def __call__(self, x):

        return (x * self.w).sum() + self.b
    
if "__main__" == __name__:
    neuron = Perceptron([21, 33])

    print(neuron.infer(1))
    print("we are Perceptron")