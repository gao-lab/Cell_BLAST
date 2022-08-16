import Cell_BLAST as cb
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("-i", "--input", dest="input", type=str, default=None)
    parser.add_argument("-o", "--output", dest="output", type=str, default=None)

    cmd_args = parser.parse_args()
    if cmd_args.input is None or cmd_args.output is None:
        raise ValueError("`-i` and `-o` must be specified!")
    cmd_args.output_path = os.path.dirname(cmd_args.output)
    if not cmd_args.output_path == '' and not os.path.exists(cmd_args.output_path):
        os.makedirs(cmd_args.output_path)

    return cmd_args

def main(cmd_args):
    cb.data.h5_to_h5ad(cmd_args.input, cmd_args.output)

if __name__ == "__main__":
    main(parse_args())
