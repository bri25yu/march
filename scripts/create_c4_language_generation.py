from argparse import ArgumentParser

from march.datasets.c4_language_generation import create_c4_for_language_generation


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--create_full", action="store_true")
    args = parser.parse_args()

    create_c4_for_language_generation(use_tiny=not args.create_full)
