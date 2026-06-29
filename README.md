It is a learning project, the implementations are inspired from karpathy's micrograd lecture.

Goal is to recreate pytorch without using external libraries. 

For testing purposes, some libraries may be in use. I hope to change that in future.

# Nature of Implementation

The network implementation is based on Andrej Karpathy's implementation of micrograd.

So it follows a graph struct(Computational graph) approach instead of network approach(used in nn playground)

In this approach, the gradient is TRANSFERED(transfered or propogated backwards) via operations for a parameter until the grad of that parameter is updated.

karpathy's approach is more like how the gradients of both operands change while performing a particular operation like addition and multiplication.

NN playgrouds approach is more formulatic and hard coded. We even require the derivate of the activation to be specified. 
Same is not true with Micrograd. Since activation functions are made up of mathematical operations, there is no need to hard-code it's derivative. 

No need to hardcode doesn't mean that it can't be or shouldn't be done. Manual derivatives are faster/ give more scope for optimization.
value.tanh implemented by karpathy used manual derivative.


|                                | Micrograd          | NN Playground            |
| ------------------------------ | ------------------ | ------------------------ |
| Node =                         | operation          | neuron                   |
| Edge =                         | dependency         | weight                   |
| Forward                        | execute ops        | propagate activations    |
| Backward                       | call `_backward()` | apply backprop equations |
| Graph changes every expression | fixed after build  |                          |

# Build guide

### Requirement
```bash
pip install build
```

### Build wheel and source distribution
```bash
python -m build
```

### Install wheel locally

Bash / Git Bash / macOS / Linux:

```bash
pip install dist/*.whl
```

OR

```bash
pip install dist/mytorch-0.1.0-py3-none-any.whl
```

> [!NOTE]  
> If you have multiple wheel files in the `dist/` directory, replace `*.whl` with the exact name of the actual wheel file you want to install.

---
log-prathm