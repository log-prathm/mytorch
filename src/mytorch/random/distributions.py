from .generator import default_generator
from .generator import Generator


def uniform(seed=None, generator= default_generator):
    """
        Every interval has same probability. 
        Ex.
        0.0-0.1 → ~10%
        0.1-0.2 → ~10%
        0.2-0.3 → ~10%
    """
    if seed is not None:
        generator = Generator(seed)
        return generator.rand()
    
    return generator.rand()

def randint(high, low, generator= default_generator):
    span = high - low

    return (
        generator.next_uint32() % span
    ) + low

def randn(shape: tuple, generator=default_generator):
    if len(shape) == 0:
        return generator.rand()
    
    return [
        randn(shape[1:], generator) for _ in range(shape[0])
    ]
    