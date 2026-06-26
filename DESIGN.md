# Terminology 

still trying to figure out what's more suitable

_children or _parent

in

```
class Tensor:
    def __init__(self, data, _children=()):
```

because the new node is created from the operations on old nodes. So _parent sounds right.

But the graphs LOOKS right as we travel from loss to input. 

karpathy famouly used _children