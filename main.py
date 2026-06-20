from core.tensor import Tensor
from nn import Perceptron
from nn.loss.square import Square 
from random import uniform
from random import randint
import argparse
from random import randn


def main():

    parser = argparse.ArgumentParser(
        description="parse the args dammin"
    )


    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed value"
    )

    args = parser.parse_args()

    if args.seed is not None:
        print(f"Seed is set to: {args.seed}")
    else:
        print(f"No seed was provided, using default seed value of 42")

    print(uniform(seed=args.seed))

if __name__ == "__main__":


    a = Tensor([[2,3]])
    b = Tensor([[4,5]])

    c = a * b
    
    c.backward()

    print(a.grad)