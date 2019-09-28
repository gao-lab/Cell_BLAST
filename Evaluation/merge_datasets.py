#! /usr/bin/env python
# by caozj
# May 23, 2018
# 9:56:38 PM

import os
import argparse
import collections
import pandas as pd
import Cell_BLAST as cb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, nargs="+")
    parser.add_argument("-m", "--mapping", dest="mapping",
                        type=str, nargs="*", default=None)
    parser.add_argument("-u", "--merge-uns-slots",
                        dest="merge_uns_slots", type=str, nargs="*", default=[])
    parser.add_argument("-o", "--output", dest="output",
                        type=str, required=True)
    cmd_args = parser.parse_args()
    cmd_args.params = argparse.Namespace(
        mapping=cmd_args.mapping,
        merge_uns_slots=cmd_args.merge_uns_slots
    )
    cmd_args.output = [cmd_args.output]
    del cmd_args.mapping, cmd_args.merge_uns_slots
    return cmd_args


def main():
    if snakemake.params.mapping is None:
        snakemake.params.mapping = [""] * len(snakemake.input)
    assert len(snakemake.input) == len(snakemake.params.mapping)

    cb.message.info("Reading data...")
    datasets = collections.OrderedDict()
    for input_file, mapping_file in zip(snakemake.input, snakemake.params.mapping):
        dataset = cb.data.ExprDataSet.read_dataset(input_file)
        if mapping_file:
            mapping = pd.read_csv(mapping_file, header=None)
            dataset = dataset.map_vars(
                mapping, map_uns_slots=snakemake.params.merge_uns_slots)
        datasets[input_file] = dataset

    merged_dataset = cb.data.ExprDataSet.merge_datasets(
        datasets, merge_uns_slots=snakemake.params.merge_uns_slots)

    cb.message.info("Saving result...")
    if not os.path.exists(os.path.dirname(snakemake.output[0])):
        os.makedirs(os.path.dirname(snakemake.output[0]))
    merged_dataset.write_dataset(snakemake.output[0])


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main()
