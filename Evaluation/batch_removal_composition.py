import argparse
import functools
import numpy as np
import pandas as pd
import Cell_BLAST as cb
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, nargs="+")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("-l", "--label", dest="label", type=str, required=True)
    parser.add_argument("-b", "--batch", dest="batch", type=str, required=True)
    cmd_args = parser.parse_args()
    cmd_args.input = argparse.Namespace(data=cmd_args.input)
    cmd_args.output = [cmd_args.output]
    cmd_args.config = argparse.Namespace(label=cmd_args.label, batch=cmd_args.batch)
    del cmd_args.label, cmd_args.batch
    return cmd_args


def main():
    output = pd.ExcelWriter(snakemake.output[0])
    for i, data_file in enumerate(snakemake.input["data"]):
        l = cb.data.read_hybrid_path("{file}//obs/{label}".format(
            file=data_file, label=snakemake.config["label"]))
        b = cb.data.read_hybrid_path("{file}//obs/{batch}".format(
            file=data_file, batch=snakemake.config["batch"]))
        mask = utils.na_mask(l)
        l, b = l[~mask], b[~mask]
        df_list = []
        for _b in np.unique(b):
            _l = l[b == _b]
            uniq, population = np.unique(_l, return_counts=True)
            proportion = population / population.sum()
            df = pd.DataFrame({
                "population": np.vectorize(str)(population),
                "proportion": np.vectorize(lambda x: "%.1f%%" % x)(proportion * 100),
                snakemake.config["label"]: uniq
            })
            df[str(_b)] = df["population"] + " (" + df["proportion"] + ")"
            del df["population"], df["proportion"]
            df_list.append(df)
        df = functools.reduce(lambda x, y: pd.merge(
            x, y, how="outer", on=snakemake.config["label"]
        ), df_list).fillna("0 (0.0%)")

        sheet_name = "group_%d" % (i + 1)
        df.to_excel(output, sheet_name=sheet_name, index=False)
    output.save()


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main()
