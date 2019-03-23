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
    parser.add_argument("-i", "--inputs", dest="inputs", type=str, nargs="+")
    parser.add_argument("-m", "--mapping", dest="mapping", type=str, nargs="*", default=None)
    parser.add_argument("-u", "--merge-uns-slots", dest="merge_uns_slots", type=str, nargs="*", default=[])
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    return parser.parse_args()


def main(cmd_args):
    if cmd_args.mapping is None:
        cmd_args.mapping = [""] * len(cmd_args.inputs)
    assert len(cmd_args.inputs) == len(cmd_args.mapping)

    cb.message.info("Reading data...")
    datasets = collections.OrderedDict()
    for input_file, mapping_file in zip(cmd_args.inputs, cmd_args.mapping):
        dataset = cb.data.ExprDataSet.read_dataset(input_file)
        if mapping_file:
            mapping = pd.read_csv(mapping_file, header=None)
            dataset = dataset.map_vars(
                mapping, map_uns_slots=cmd_args.merge_uns_slots)
        datasets[input_file] = dataset

    merged_dataset = cb.data.ExprDataSet.merge_datasets(
        datasets, merge_uns_slots=cmd_args.merge_uns_slots)

    cb.message.info("Saving result...")
    if not os.path.exists(os.path.dirname(cmd_args.output)):
        os.makedirs(os.path.dirname(cmd_args.output))
    merged_dataset.write_dataset(cmd_args.output)


if __name__ == "__main__":
    main(parse_args())
    print("Done!")
