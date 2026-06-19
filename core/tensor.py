class Tensor:
    def __init__(self, lst):
        self.Tensor = lst
        # self.shape = self.size()

    def __call__(self):
        return self.Tensor
    
    @property
    def shape(self) -> tuple:
        """Returns the dimensions of the Tensor as a tuple"""
        dims = []
        current = self.Tensor

        while isinstance(current, list):
            dims.append(len(current))
            if len(current) == 0:
                break
            current = current[0]
        return tuple(dims)
    
    def size(self) -> tuple:
        """Returns the dimensions of the Tensor as a tuple"""
        dims = []
        current = self.Tensor

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
            return other.Tensor
        return other
    
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        """Handles element wise Tensor substraction"""
        other_data = self._scalar_or_Tensor(other)
        if isinstance(other, Tensor) and self.shape != other.shape:
            raise ValueError(f"Shape mismatch: cannot substract: {self.shape} and {other.shape}")

        if not isinstance(self.Tensor, list):
            return Tensor(self.Tensor - other_data)

        if not isinstance(other_data, list):
            result_data = self._elementwise_op(self.Tensor, self._broadcast_like(self.Tensor, other_data), lambda x, y: x - y)
        else:
            result_data = self._elementwise_op(self.Tensor, other_data, lambda x, y: x-y)
        return Tensor(result_data)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        """Handles element wise Tensor addition"""
        other_data = self._scalar_or_Tensor(other)
        if isinstance(other, Tensor) and self.shape != other.shape:
            raise ValueError(f"Shape mismatch: Cannot Add: {self.shape} and {other.shape}")

        if not isinstance(self.Tensor, list):
            return Tensor(self.Tensor + other_data)

        if not isinstance(other_data, list):
            result_data = self._elementwise_op(self.Tensor, self._broadcast_like(self.Tensor, other_data), lambda x, y: x + y)
        else:
            result_data = self._elementwise_op(self.Tensor, other_data, lambda x, y: x+y)
        return Tensor(result_data)

    def __radd__(self, other):
        return self + other

    def _broadcast_like(self, template, value):
        if not isinstance(template, list):
            return value
        return [self._broadcast_like(template[0], value) for _ in template]

    def __mul__(self, other):
        other_data = self._scalar_or_Tensor(other)

        if isinstance(other, Tensor) and self.shape != other.shape:
            raise ValueError(f"Shape mismatch: Cannot Multiply: {self.shape} and {other.shape}")

        if not isinstance(self.Tensor, list):
            return Tensor(self.Tensor * other_data)

        if not isinstance(other_data, list):
            result_data = self._elementwise_op(self.Tensor, self._broadcast_like(self.Tensor, other_data), lambda x, y: x * y)
        else:
            result_data = self._elementwise_op(self.Tensor, other_data, lambda x, y: x * y)
        return Tensor(result_data)
    
    def __pow__(self, exponent: int):
        """return Tensor powered"""

        result_data = self._elementwise_op(self.Tensor, self.Tensor, lambda x, y: x**exponent)

        return result_data

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        """Helper to print the Tensor cleanly."""
        return f"Tensor({self.Tensor})"

    def __getitem__(self, index):
        """Allows bracket indexing and slicing: Tensor[index]"""
        result = self.Tensor[index]

        if isinstance(result, list):
            return Tensor(result)
        
        return result
if "__main__" == __name__: 
    app = Tensor([[1,2,3], [2,3,4]])
    print(app[0])
