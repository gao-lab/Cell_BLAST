import os
import argparse
import numpy as np
import Cell_BLAST as cb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", dest="data", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("--x-low", dest="x_low", type=float, default=0.1)
    parser.add_argument("--x-high", dest="x_high", type=float, default=8.0)
    parser.add_argument("--y-low", dest="y_low", type=float, default=1.0)
    parser.add_argument("--y-high", dest="y_high", type=float, default=np.inf)
    cmd_args = parser.parse_args()
    os.makedirs(os.path.dirname(cmd_args.output), exist_ok=True)
    return cmd_args


def main(cmd_args):
    ds = cb.data.ExprDataSet.read_dataset(cmd_args.data)
    selected, _ = ds.find_variable_genes(
        x_low_cutoff=cmd_args.x_low,
        x_high_cutoff=cmd_args.x_high,
        y_low_cutoff=cmd_args.y_low,
        y_high_cutoff=cmd_args.y_high,
    )
    np.savetxt(cmd_args.output, np.array(selected), fmt="%s")


if __name__ == "__main__":
    main(parse_args())
    print("Done!")
