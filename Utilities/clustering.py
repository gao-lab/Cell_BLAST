#! /usr/bin/env python

import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append("..")
import Cell_BLAST.message
import Cell_BLAST.data
import Cell_BLAST.metrics
import Cell_BLAST.utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cluster", dest="cluster",
                        type=str, required=True)
    parser.add_argument("-r", "--reference", dest="reference",
                        type=str, required=True)
    parser.add_argument("-f", "--confusion", dest="confusion",
                        default=False, action="store_true")
    parser.add_argument("-t", "--completeness", dest="completeness",
                        default=False, action="store_true")
    parser.add_argument("-g", "--homogeneity", dest="homogeneity",
                        default=False, action="store_true")
    parser.add_argument("-a", "--ari", dest="ari",
                        default=False, action="store_true")
    parser.add_argument("-n", "--nmi", dest="nmi",
                        default=False, action="store_true")
    parser.add_argument("-k", "--kappa", dest="kappa",
                        default=False, action="store_true")
    parser.add_argument("-u", "--accuracy", dest="accuracy",
                        default=False, action="store_true")
    parser.add_argument("--cell-type-dag", dest="cell_type_dag",
                        type=str, default=None)
    parser.add_argument("-s", "--save-confusion", dest="save_confusion",
                        type=str, default=None)
    return parser.parse_args()


def main():
    cmd_args = parse_args()
    to_str = np.vectorize(str)

    cb.message.info("Reading data...")
    c = to_str(cb.data.read_hybrid_path(cmd_args.cluster))
    r = to_str(cb.data.read_hybrid_path(cmd_args.reference))

    cov = cb.metrics.coverage(c)
    print("Coverage = %f" % cov)

    if cmd_args.confusion:
        cm = cb.metrics.confusion_matrix(r, c)
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            print(cm.loc[:, cm.sum(axis=0) != 0])
        if cmd_args.save_confusion:
            cm.to_csv(cmd_args.save_confusion)

    if not (cmd_args.completeness or cmd_args.homogeneity or
            cmd_args.ari or cmd_args.nmi or
            cmd_args.kappa or cmd_args.accuracy) or cov == 0:
        sys.exit(1)

    if cmd_args.completeness:
        print("Completeness = %f" % cb.metrics.completeness(r, c))
    if cmd_args.homogeneity:
        print("Homogeneity = %f" % cb.metrics.homogeneity(r, c))
    if cmd_args.nmi:
        print("NMI = %f" % cb.metrics.nmi(r, c))
    if cmd_args.ari:
        print("ARI = %f" % cb.metrics.ari(r, c))
    if cmd_args.kappa:
        print("Cohen's Kappa = %f" % cb.metrics.kappa(r, c))
    if cmd_args.accuracy:
        dag = cb.utils.CellTypeDAG.load(cmd_args.cell_type_dag) \
            if cmd_args.cell_type_dag is not None else None
        print("Accuracy = %f" % cb.metrics.accuracy(
            r, c, similarity=dag.is_ancestor_of
            if dag is not None else None))


if __name__ == "__main__":
    main()
    cb.message.info("Done!")
