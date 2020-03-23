#! /usr/bin/env python
# by caozj
# Jan 13, 2020
# 3:12:30 PM


import argparse
import time
import umap
import Cell_BLAST as cb


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input", type=str, required=True)
parser.add_argument("-o", "--output", dest="output", type=str, required=True)
parser.add_argument("-d", "--dim", dest="dim", type=int, default=2)
parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
cmd_args = parser.parse_args()

# Read data
print("Reading data...")
x = cb.data.read_hybrid_path(cmd_args.input)

print("Running UMAP...")
start_time = time.time()
z = umap.UMAP(
    n_components=cmd_args.dim, random_state=cmd_args.seed
).fit_transform(x)
elapsed_time = time.time() - start_time

print("Saving result...")
cb.data.write_hybrid_path(z, "%s//latent" % cmd_args.output)
cb.data.write_hybrid_path(elapsed_time, "%s//time" % cmd_args.output)

print("Done!")
