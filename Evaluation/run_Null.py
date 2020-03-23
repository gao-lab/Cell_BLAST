#! /usr/bin/env python
# by caozj
# Jan 23, 2020
# 11:40:37 AM

import argparse
import numpy as np
import Cell_BLAST as cb
import utils


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input", type=str, required=True)
parser.add_argument("-o", "--output", dest="output", type=str, required=True)
parser.add_argument("-g", "--genes", dest="genes", type=str, default=None)
parser.add_argument("--clean", dest="clean", type=str, default=None)
cmd_args = parser.parse_args()

# Read data
print("Reading data...")
x = cb.data.ExprDataSet.read_dataset(cmd_args.input).normalize()
if cmd_args.clean:
    x = utils.clean_dataset(x, cmd_args.clean)
if cmd_args.genes is not None:
    x = cb.utils.densify(np.log1p(x[:, x.uns[cmd_args.genes]].exprs))

# Save result
cb.data.write_hybrid_path(x, "%s//exprs" % cmd_args.output)
cb.data.write_hybrid_path(0, "%s//time" % cmd_args.output)

print("Done!")
