import collections
import argparse
import numpy as np
import pandas as pd
import parse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, nargs="+")
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("--pattern", dest="pattern", type=str, required=True)
    parser.add_argument("--selected", dest="selected", type=str, required=True)
    cmd_args = parser.parse_args()
    cmd_args.input.append(cmd_args.selected)
    cmd_args.output = [cmd_args.output]
    cmd_args.params = argparse.Namespace(pattern=cmd_args.pattern)
    del cmd_args.selected, cmd_args.pattern
    return cmd_args


def main():
    df_selected = pd.read_csv(snakemake.input["selected"], index_col=0)
    tree = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(
                list
            )
        )
    )
    for item in set(snakemake.input["data"]):
        d = parse.parse(snakemake.params.pattern, item).named
        for threshold in df_selected.loc[[d["method"].replace("_", " ")], "threshold"]:
            tree[d["group"]][d["method"]][threshold].append(pd.read_excel(
                item, sheet_name=str(threshold), index_col=0))

    output = pd.ExcelWriter(snakemake.output[0])
    for group in sorted(tree.keys()):
        df = None
        for method in snakemake.config["method"]:  # For correct ordering
            for threshold in sorted(tree[group][method].keys()):
                l = tree[group][method][threshold]
                for i in range(1, len(l)):
                    assert np.all(l[0].index == l[i].index)
                _df = pd.concat(
                    l, axis=0, keys=np.arange(len(l))
                ).mean(level=1).reset_index()
                _df.columns = np.vectorize(
                    lambda x: "{method}".format(
                        method=method
                    ) if x == "accuracy" else "cell_ontology_class" if x == "index" else x
                )(_df.columns)
                if df is None:
                    df = _df
                else:
                    df = df.merge(_df)
        df.to_excel(output, sheet_name=group, index=False)
        output.save()


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main()
