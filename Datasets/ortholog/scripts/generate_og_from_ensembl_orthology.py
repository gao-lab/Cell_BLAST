#! /usr/bin/env python
# by caozj
# Sep 7, 2018
# 12:46:00 PM


import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--taxids", dest="taxids", type=int, nargs=2)
parser.add_argument("-n", "--use-name", dest="use_name", default=False, action="store_true")
cmd_args = parser.parse_args()

df = pd.read_csv(os.path.join(
    "..", "Ensembl", "orthology", "%d_%d.csv" % tuple(cmd_args.taxids)
), header=None)
if cmd_args.use_name:
    df = df.iloc[:, [1, 3]]
else:
    df = df.iloc[:, [0, 2]]
df.columns = [0, 1]
genes = {
    cmd_args.taxids[0]: np.unique(df[0]),
    cmd_args.taxids[1]: np.unique(df[1])
}
combined_genes = np.concatenate(list(genes.values()))
gene_lut = {combined_genes[i]: i for i in range(len(combined_genes))}

graph = sp.lil_matrix((len(combined_genes), len(combined_genes)), dtype=np.int8)
for i, row in df.iterrows():
    graph[gene_lut[row[0]], gene_lut[row[1]]] = 1
    graph[gene_lut[row[1]], gene_lut[row[0]]] = 1
components = sp.csgraph.connected_components(graph, directed=False)[1]

target_path = os.path.join("..", "Ensembl", "orthology", "%d_%d" % tuple(cmd_args.taxids))
if not os.path.exists(target_path):
    os.makedirs(target_path)
for taxid in cmd_args.taxids:
    with open(os.path.join(target_path, "%d.csv" % taxid), "w") as f:
        for gene in genes[taxid]:
            f.write("%s,OG:%06d\n" % (gene, components[gene_lut[gene]]))

print("Done!")
