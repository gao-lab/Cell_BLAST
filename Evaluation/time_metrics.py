import os
import argparse
import json
import Cell_BLAST as cb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    cmd_args = parser.parse_args()
    cmd_args.output = [cmd_args.output]
    cmd_args.input = [cmd_args.input]
    return cmd_args


def main():
    if not os.path.exists(os.path.dirname(snakemake.output[0])):
        os.makedirs(os.path.dirname(snakemake.output[0]))
    with open(snakemake.output[0], "w") as f:
        json.dump({
            "time": cb.data.read_hybrid_path(
                "//".join([snakemake.input[0], "time"])
            )
        }, f)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main()
