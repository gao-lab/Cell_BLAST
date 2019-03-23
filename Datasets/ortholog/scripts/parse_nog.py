#! /usr/bin/env python
# by caozj
# Jul 20, 2018
# 2:53:22 PM

import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

NOG = "biNOG"

print("Reading table...")
input_file = "../%s.members.tsv.gz" % NOG
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
            d[taxid] = {}
        if gene not in d[taxid]:
            d[taxid][gene] = [og]
        else:
            d[taxid][gene].append(og)

print("Saving result...")
if not os.path.exists("../%s.members" % NOG):
    os.makedirs("../%s.members" % NOG)
def _save(taxid):
    with open("../%s.members/%s.txt" % (NOG, taxid), "w") as f:
        for gene in d[taxid]:
            for og in d[taxid][gene]:
                f.write("%s\t%s\n" % (gene, og))
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(_save, d.keys())

print("Done!")
