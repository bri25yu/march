from argparse import ArgumentParser

from march.utils import print_most_recent_runs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_runs", type=int)
    args = parser.parse_args()

    print_most_recent_runs(num_runs=args.num_runs)
