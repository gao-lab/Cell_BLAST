#!/usr/bin/env python

import sys
import os
import argparse
import numpy as np
import plotly.io
import Cell_BLAST as cb

sys.path.insert(0, "../Evaluation")
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--true", dest="true", type=str, nargs="+")
    parser.add_argument("-p", "--pred", dest="pred", type=str, nargs="+")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("--clean-true", dest="clean_true", default=False, action="store_true")
    parser.add_argument("--tint-cutoff", dest="tint_cutoff", type=int, default=1)
    parser.add_argument("--title", dest="title", type=str, default="Sankey")
    parser.add_argument("--width", dest="width", type=float, default=500.0)
    parser.add_argument("--height", dest="height", type=float, default=500.0)
    cmd_args = parser.parse_args()
    assert len(cmd_args.true) == len(cmd_args.pred)
    return cmd_args


def main(cmd_args):
    true, pred = [], []
    for item in cmd_args.true:
        true.append(cb.data.read_hybrid_path(item))
    for item in cmd_args.pred:
        pred.append(cb.data.read_hybrid_path(item))
    true, pred = np.concatenate(true), np.concatenate(pred)
    if cmd_args.clean_true:
        mask = utils.na_mask(true)
        true = true[~mask]
    fig = cb.blast.sankey(
        true, pred, title=cmd_args.title,
        width=cmd_args.width, height=cmd_args.height,
        tint_cutoff=cmd_args.tint_cutoff, suppress_plot=True
    )
    if not os.path.exists(os.path.dirname(cmd_args.output)):
        os.makedirs(os.path.dirname(cmd_args.output))
    plotly.io.write_image(fig, cmd_args.output)


if __name__ == "__main__":
    main(parse_args())
