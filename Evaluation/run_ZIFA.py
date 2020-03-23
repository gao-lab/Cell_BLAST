#! /usr/bin/env python
# by caozj
# 29 Jan 2018
# 4:33:45 PM


import argparse
import time
import numpy as np
import scipy.sparse as spsp
import Cell_BLAST as cb
from ZIFA import ZIFA
import utils


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input", type=str, required=True)
parser.add_argument("-o", "--output", dest="output", type=str, required=True)
parser.add_argument("-g", "--genes", dest="genes", type=str, default=None)
parser.add_argument("-d", "--dim", dest="dim", type=int, default=2)
parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
parser.add_argument("--clean", dest="clean", type=str, default=None)
cmd_args = parser.parse_args()

# Read data
print("Reading data...")
x = cb.data.ExprDataSet.read_dataset(cmd_args.input).normalize()
if cmd_args.clean:
    x = utils.clean_dataset(x, cmd_args.clean)
if cmd_args.genes is not None:
    x = x[:, x.uns[cmd_args.genes]].exprs

# Run ZIFA
if cmd_args.seed is not None:
    np.random.seed(cmd_args.seed)
start_time = time.time()
x = np.log1p(x)
if spsp.issparse(x):
    x = x.toarray()
z, _ = ZIFA.fitModel(x, cmd_args.dim)
elapsed_time = time.time() - start_time

# Save result
cb.data.write_hybrid_path(z, "%s//latent" % cmd_args.output)
cb.data.write_hybrid_path(elapsed_time, "%s//time" % cmd_args.output)

print("Done!")
