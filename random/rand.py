import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.tensor import Tensor

class rand:

    def generate(self, dim: tuple) -> Tensor:
        """Generate random Tensor with tuple dimensions"""
        try:
            if not isinstance(dim, tuple):
                    dim = tuple(dim)
        except:
             print("Invalid dimensions, please enter a tuple")

        try:
             import numpy as np
        except ImportError:
             print("Numpy library not available")

        rng = np.random.default_rng()
        int_matrix = rng.integers()