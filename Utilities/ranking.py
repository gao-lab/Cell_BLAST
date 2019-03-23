#!/usr/bin/env python

import sys
import argparse

sys.path.append("..")
import Cell_BLAST.data
import Cell_BLAST.utils
import Cell_BLAST.metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hits", dest="hits", type=str, required=True)
    parser.add_argument("--ref", dest="ref", type=str, required=True)
    parser.add_argument("--cell-type-dag", dest="cell_type_dag", type=str, default=None)
    return parser.parse_args()


def main():
    cmd_args = parse_args()
    hits = Cell_BLAST.data.read_hybrid_path(cmd_args.hits)
    ref = Cell_BLAST.data.read_hybrid_path(cmd_args.ref)
    if cmd_args.cell_type_dag:
        dag = Cell_BLAST.utils.CellTypeDAG.load(cmd_args.cell_type_dag)
        print("MAFP = %.3f" % Cell_BLAST.metrics.mean_average_precision(ref, hits, dag.similarity))
    else:
        print("MAP = %.3f" % Cell_BLAST.metrics.mean_average_precision(ref, hits))


if __name__ == "__main__":
    main()
    print("Done!")
