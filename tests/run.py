import argparse
from numericals import singular
from numericals import vector

def handle_arguments():
    parser = argparse.ArgumentParser(
        prog="test",
        description="tests the mytorch library",
    )

    parser.add_argument("--singular", action="store_true", help="Run only singular addition tests")
    parser.add_argument("--vector", action="store_true", help="Run only vector addition tests")

    return parser.parse_args()


def main():
    args = handle_arguments()

    if args.singular:
        singular()
    if args.vector:
        vector()


if __name__ == "__main__":
    main()