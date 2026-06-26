It is a learning project, the implementations are inspired from karpathy's micrograd lecture.

Goal is to recreate pytorch without using external libraries. 

For testing purposes, some libraries may be in use. I hope to change that in future.

# Nature of Implementation

The network implementation is based on Andrej Karpathy's implementation of micrograd.

So it follows a graph struct(Computational graph) approach instead of network approach(used in nn playground)


|                                | Micrograd          | NN Playground            |
| ------------------------------ | ------------------ | ------------------------ |
| Node =                         | operation          | neuron                   |
| Edge =                         | dependency         | weight                   |
| Forward                        | execute ops        | propagate activations    |
| Backward                       | call `_backward()` | apply backprop equations |
| Graph changes every expression | fixed after build  |                          |


---
log-prathm