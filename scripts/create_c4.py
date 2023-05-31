from argparse import ArgumentParser

from march.datasets.c4 import create_span_corrupted_c4


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--create_full", action="store_true")
    args = parser.parse_args()

    create_span_corrupted_c4(use_tiny=not args.create_full)
