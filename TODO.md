# Recursive operations introduces abnomality in operations

1. Shape property or size function implementation

We are recursively calling first entity of the list until the element noticed is not a list.
This is good enough but for Tensors with unequal elements, the size/shape is wrong.

2. same recursive nature is present in __mul__, __add__ and __sub__ operation

standard shapes such as [2,3], [3,3] etc works.

But tensors like [[1,2], [2,3,4]] will return [2,2] because of first entry.






