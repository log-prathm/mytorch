class Optimizer:

    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def zero_grad(self):

        for p in self.params:

            p.grad = (
                p._zeros_like(
                    p.data
                )
            )
        
    def step(self):
        raise NotImplementedError
    
class SGD(Optimizer):
    """performing w = w - lr*dw"""
    def step(self):

        for p in self.params:

            p.data = p._elementwise_op(
                p.data,
                p.grad,
                lambda d, g: d - self.lr*g
            )