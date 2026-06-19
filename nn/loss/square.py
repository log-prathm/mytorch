from core.tensor import Tensor

class Square:


    def __call__(self, y: Tensor, _y:Tensor):
        return (y - _y)**2