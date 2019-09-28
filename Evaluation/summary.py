#!/usr/bin/env python

import json
import argparse
import pandas as pd
import parse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, nargs="+")
    parser.add_argument("-p", "--pattern", dest="pattern", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    cmd_args = parser.parse_args()
    cmd_args.params = argparse.Namespace(pattern=cmd_args.pattern)
    del cmd_args.pattern
    return cmd_args


def main():
    df = []
    for item in set(snakemake.input):
        with open(item, "r") as f:
            performance = json.load(f)
        if isinstance(performance, dict):
            performance = [performance]
        for _performance in performance:
            _performance.update(parse.parse(snakemake.params.pattern, item).named)
            df.append(_performance)
    df = pd.DataFrame.from_records(df)
    df.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main()
