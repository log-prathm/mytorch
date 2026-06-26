from core.tensor import Tensor
import argparse
from nn.loss import mse
from nn import Layer


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
    x = Tensor([x for x in range(10)])
    y = ([2*y for y in range(10)])
    model = Layer(1, 1)
    epochs = 10
    model.zero_grad()
    train_data = zip(x,y)
    print(f"model para before: {model.parameters()}")

    for epoch in range(epochs):

        for X, Y in train_data:
            y_preds = model(X)

            loss = mse(y_preds, Y)
            loss.backward()

        print(loss)
    print(f'model parameters after: {model.parameters()}')