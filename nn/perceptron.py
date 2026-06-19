import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.tensor import Tensor

class Perceptron:

    def __init__(self, args: Tensor):
        self.bias = args[0]
        self.weight = args[1]

    def infer(self, X: Tensor) -> Tensor:
        output = self.bias + self.weight*X

        if not isinstance(output, Tensor):
            return Tensor(output)
        
        return output
    
if "__main__" == __name__:
    neuron = Perceptron([21, 33])

    print(neuron.infer(1))
    print("we are Perceptron")