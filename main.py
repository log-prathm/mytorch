from core.tensor import Tensor
from nn import Perceptron
from nn.loss.square import Square 


y = Tensor([21, 33])
_y = Tensor([22.5, 34])

loss_fn = Square()
print(loss_fn(y, _y))
