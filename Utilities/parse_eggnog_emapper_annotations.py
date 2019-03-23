#! /usr/bin/env python
# by caozj
# Aug 1, 2018
# 5:28:07 PM

import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input", type=str, required=True)
parser.add_argument("-n", "--nog", dest="nog", type=str, required=True)
parser.add_argument("-p", "--prefix", dest="prefix", type=str, required=True)
parser.add_argument("-o", "--output", dest="output", type=str, required=True)
cmd_args = parser.parse_args()

# cmd_args.nog = "@meNOG"
# cmd_args.prefix = "ENOG41"


@np.vectorize
def extract_nog(ann):
    return ",".join(
        cmd_args.prefix + item.replace(cmd_args.nog, "")
        for item in filter(
            lambda x: x.find(cmd_args.nog) > -1,
            ann.split(",")
        )
    )


df = pd.read_table(cmd_args.input, header=None)
df = df.iloc[:, [0, 9]]
df[9] = pd.Series(extract_nog(df[9].values))
mask = df[9].values != ""
df = df.iloc[mask, :]

df.to_csv(cmd_args.output, sep="\t", index=False, header=False)
print("Done!")
