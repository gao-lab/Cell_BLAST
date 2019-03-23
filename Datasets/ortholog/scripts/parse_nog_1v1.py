#! /usr/bin/env python
# by caozj
# Jul 20, 2018
# 2:53:22 PM

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

NOG = "meNOG"

print("Reading table...")
input_file = "%s.members.tsv.gz" % NOG
df = pd.read_table(input_file, header=None)
# df.columns = ["dataset", "id", "num_proteins", "num_species",
#               "category", "members"]

print("Scanning table...")
d = {}
for i in tqdm(range(df.shape[0]), unit="rows"):
    members = df.iloc[i, -1]
    members = members.split(",")
    for member in members:
        member_split = member.split(".")
        taxid = member_split[0]
        gene = ".".join(member_split[1:])
        og = df.iloc[i, 1]
        if taxid not in d:
            d[taxid] = {
                "gene": [],
                "og": []
            }
        d[taxid]["gene"].append(gene)
        d[taxid]["og"].append(og)

print("Building 1v1 map...")
for taxid in tqdm(d.keys(), unit="species"):
    unique_genes = np.unique(d[taxid]["gene"])
    unique_ogs = np.unique(d[taxid]["og"])
    matrix = np.zeros((len(unique_genes), len(unique_ogs)), dtype=np.int8)
    gene_idx, og_idx = {}, {}
    for i in range(len(unique_genes)):
        gene_idx[unique_genes[i]] = i
    for i in range(len(unique_ogs)):
        og_idx[unique_ogs[i]] = i
    for i in range(len(d[taxid]["gene"])):
        matrix[
            gene_idx[d[taxid]["gene"][i]],
            og_idx[d[taxid]["og"][i]],
        ] = 1
    del d[taxid]["gene"], d[taxid]["og"]
    while True:
        cmask = matrix.sum(axis=0) == 1
        rmask = matrix.sum(axis=1) == 1
        if np.all(cmask) and np.all(rmask):
            break
        if not np.all(cmask):
            matrix = matrix[:, cmask]
        if not np.all(rmask):
            matrix = matrix[rmask, :]
    assert matrix.shape[0] == matrix.shape[1]
    cmap = unique_genes[matrix.argmax(axis=0)]
    rmap = unique_ogs[matrix.argmax(axis=1)]
    d[taxid]["og2g"] = {}
    for i in range(matrix.shape[1]):
        d[taxid]["og2g"][unique_ogs[i]] = cmap[i]
    d[taxid]["g2og"] = {}
    for i in range(matrix.shape[0]):
        d[taxid]["g2og"][unique_genes[i]] = rmap[i]
    del matrix

print("Sanity check...")
for tax_d in tqdm(d.values(), unit="species"):
    assert len(tax_d["g2og"]) == len(tax_d["og2g"])
    for item in tax_d["g2og"]:
        assert item == tax_d["og2g"][tax_d["g2og"][item]]
    for item in tax_d["og2g"]:
        assert item == tax_d["g2og"][tax_d["og2g"][item]]

print("Saving result...")
if not os.path.exists("%s.members.1v1" % NOG):
    os.makedirs("%s.members.1v1" % NOG)
def _save(taxid):
    with open("%s.members.1v1/%s_g2og.pkl" % (NOG, taxid), "wb") as f:
        pickle.dump(d[taxid]["g2og"], f)
    with open("%s.members.1v1/%s_og2g.pkl" % (NOG, taxid), "wb") as f:
        pickle.dump(d[taxid]["og2g"], f)
with ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(_save, d.keys())

print("Done!")
