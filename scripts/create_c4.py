from argparse import ArgumentParser

from march.datasets.c4 import create_span_corrupted_c4


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use_tiny", action="store_true", default=True)
    args = parser.parse_args()

    create_span_corrupted_c4(use_tiny=args.use_tiny)
