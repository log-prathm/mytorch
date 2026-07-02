from core import Tensor

a = Tensor(-2.0)
b = Tensor(3.0)
d = a * b
e = a + b
f = d * e
f.backward()

print("a",a.grad)
print("e",e.grad)
print("f",f.grad)