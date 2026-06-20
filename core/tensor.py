from random.distributions import randn

class Tensor:
    def __init__(self, data, _parent=()):
        self.data = data
        self._backward = lambda: None
        self._prev = set(_parent)
        self.grad = self._zeros_like(data)

    def __call__(self):
        return self.data
    
    @property
    def shape(self) -> tuple:
        """Returns the dimensions of the Tensor as a tuple"""
        dims = []
        current = self.data

        while isinstance(current, list):
            dims.append(len(current))
            if len(current) == 0:
                break
            current = current[0]
        return tuple(dims)
    
    def size(self) -> tuple:
        """Returns the dimensions of the Tensor as a tuple"""
        dims = []
        current = self.data

        while isinstance(current, list):
            dims.append(len(current))
            if len(current) == 0:
                break
            current = current[0]
        return tuple(dims)
    
    def _elementwise_op(self, a, b, op):
        """Recursively applies element wise operations on two Tensors"""

        if not isinstance(a, list):
            # base case
            return op(a, b)
        
        return [self._elementwise_op(item_a, item_b, op) for item_a, item_b in zip(a, b)]

    def _scalar_or_Tensor(self, other):
        if isinstance(other, Tensor):
            return other.data
        return other
    
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        """Handles element wise Tensor substraction"""
        other_data = self._scalar_or_Tensor(other)
        if isinstance(other, Tensor) and self.shape != other.shape:
            raise ValueError(f"Shape mismatch: cannot substract: {self.shape} and {other.shape}")

        if not isinstance(other_data, list):
            result_data = self._elementwise_op(self.data, self._broadcast_like(self.data, other_data), lambda x, y: x - y)
        else:
            result_data = self._elementwise_op(self.data, other_data, lambda x, y: x-y)
        out = Tensor(
            result_data,
            (self, other) if isinstance(other, Tensor) else (self, )
        )
        return out
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        """Handles element wise Tensor addition"""
        other_data = self._scalar_or_Tensor(other)

        if isinstance(other, Tensor) and self.shape != other.shape:
            raise ValueError(f"Shape mismatch: Cannot Add: {self.shape} and {other.shape}")

        if not isinstance(self.data, list):
            result_data = self._elementwise_op(
                self.data,
                self._broadcast_like(self.data, other_data),
                lambda x, y: x+y
            )
        else:
            result_data = self._elementwise_op(
                self.data, 
                other_data, 
                lambda x, y: x+y
            )

        out = Tensor(
            result_data,
            (self, other) if isinstance(other, Tensor) else (self,)
        )
        def _backward():
            self.grad = self._elementwise_op(
                self.grad,
                out.grad,
                lambda g, og: g + og
            )

            if isinstance(other, Tensor):
                other.grad = self._elementwise_op(
                    other.grad,
                    out.grad,
                    lambda g, og: g + og
                )
        out._backward = _backward

        return out

    def __radd__(self, other):
        """
            3 + x -> int.__add__(x) won't work because x isn't int.
            so radd does 3 + x => x + 3
            so x.__add__(3) will workout
        """
        return self + other

    def _broadcast_like(self, template, value):
        """
            If does -> Tensor([[1, 2], [3, 4]]) + 10
            converting 10 into [[10, 10], [10, 10]] so the addition works out
        """
        if not isinstance(template, list):
            return value
        return [self._broadcast_like(template[0], value) for _ in template]
    
    def _zeros_like(self, data):
        if not isinstance(data, list):
            return 0
        return [self._zeros_like(x) for x in data]

    def _ones_like(self, data):
        if not isinstance(data, list):
            return 1
        return [self._ones_like(x) for x in data]
    
    def __mul__(self, other):
        other_data = self._scalar_or_Tensor(other)

        if isinstance(other, Tensor) and self.shape != other.shape:
            raise ValueError(f"Shape mismatch: Cannot Multiply: {self.shape} and {other.shape}")


        if not isinstance(other_data, list):
            result_data = self._elementwise_op(self.data, self._broadcast_like(self.data, other_data), lambda x, y: x * y)
        else:
            result_data = self._elementwise_op(self.data, other_data, lambda x, y: x * y)

        out = Tensor(
            result_data,
            (self, other) if isinstance(other, Tensor) else (self, )
        )
        def _backward():
            self.grad = self._elementwise_op(
                self.grad,
                self._elementwise_op(
                    other.data, out.grad,
                    lambda x, y: x*y
                ),
                lambda g, ng: g+ng
            )  

        out._backward = _backward

        return out  
    def __pow__(self, exponent: int):
        """return Tensor powered"""

        result_data = self._elementwise_op(self.data, self.data, lambda x, y: x**exponent)

        return result_data

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        """Helper to print the Tensor cleanly."""
        return f"Tensor({self.data})"

    def __getitem__(self, index):
        """Allows bracket indexing and slicing: Tensor[index]"""
        result = self.data[index]

        if isinstance(result, list):
            return Tensor(result)
        
        return result
    
    def randn(self, size: tuple) -> 'Tensor':
        return Tensor(randn(size))

    def backward(self):

        topo = []
        visited= set()

        def build(v):
            if v not in visited:
                visited.add(v)

                for parent in v._prev:
                    build(parent)

                topo.append(v)

        build(self)

        self.grad = self._ones_like(self.data)

        for node in reversed(topo):
            node._backward()

if "__main__" == __name__: 
    app = Tensor([[1,2,3], [2,3,4]])
    print(app[0])
