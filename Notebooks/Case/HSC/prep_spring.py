#!/usr/bin/env python
# Run with SPRING environment

import sys
import argparse
import numpy as np
import scipy.sparse

sys.path.append("./SPRING")
import preprocessing_python


parser = argparse.ArgumentParser()
parser.add_argument("-e", dest="expr", type=str, required=True)
parser.add_argument("-d", dest="dist", type=str, required=True)
parser.add_argument("-g", dest="gene", type=str, required=True)
parser.add_argument("-k", dest="k", type=int, required=True)
parser.add_argument("-o", dest="output", type=str, required=True)
cmd_args = parser.parse_args()

expr = np.load(cmd_args.expr, allow_pickle=True)
dist = np.load(cmd_args.dist, allow_pickle=True)
gene = np.load(cmd_args.gene, allow_pickle=True)

preprocessing_python.save_spring_dir(expr, dist, cmd_args.k, gene.tolist(), cmd_args.output)
