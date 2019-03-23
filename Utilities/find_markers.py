#! /usr/bin/env python
# by caozj
# May 16, 2018
# 7:31:00 PM

"""
This script find marker genes based on certain grouping
"""

import os
import sys
import time
import argparse
import numpy as np
import sklearn

sys.path.append("..")
import Cell_BLAST.data
import Cell_BLAST.utils
from markers import fast_markers


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", dest="data", type=str, required=True)
parser.add_argument("-g", "--grouping", dest="grouping",
                    type=str, required=True)
parser.add_argument("-v", "--var-names", dest="var_names",
                    type=str, default=None)
parser.add_argument("-a", "--alternative", dest="alternative",
                    type=str, default="greater",
                    choices=["two-sided", "greater", "less"])
parser.add_argument("-o", "--output-path", dest="output_path",
                    type=str, required=True)
parser.add_argument("-j", "--n-jobs", dest="n_jobs", type=int, default=1)
parser.add_argument("-f", "--filters", dest="filters", type=str, nargs="*")
cmd_args = parser.parse_args()

if cmd_args.data.find("//") == -1:
    data = Cell_BLAST.data.ExprDataSet.read_dataset(cmd_args.data)
else:
    assert cmd_args.var_names is not None
    data = Cell_BLAST.utils.dotdict(
        exprs=Cell_BLAST.data.read_hybrid_path(cmd_args.data),
        var_names=Cell_BLAST.data.read_hybrid_path(cmd_args.var_names)
    )
    data.exprs = sklearn.preprocessing.normalize(
        data.exprs, norm="l1", copy=False
    ) * 10000

if cmd_args.filters:
    filter_mask = np.ones(data.shape[0]).astype(bool)
    filter_name, filter_vals, filter_opts = None, None, []
    for item in cmd_args.filters:
        if item.startswith("@"):
            if filter_name is not None:  # Not first filter
                assert filter_vals is not None and filter_opts is not None
                filter_mask_tmp = np.zeros(data.shape[0]).astype(bool)
                for filter_opt in filter_opts:
                    filter_mask_tmp_tmp = filter_vals == filter_opt
                    assert np.any(filter_mask_tmp_tmp)
                    filter_mask_tmp = np.logical_or(
                        filter_mask_tmp, filter_mask_tmp_tmp
                    )
                filter_mask = np.logical_and(
                    filter_mask, filter_mask_tmp
                )
            filter_name = item.strip("@")
            filter_split = filter_name.split("//")
            filter_opts = []
            if len(filter_split) == 1:
                filter_vals = data.obs[filter_name]
            elif len(filter_split) == 2:
                filter_vals = Cell_BLAST.data.read_hybrid_path(filter_name)
        else:
            assert filter_name is not None
            filter_opts.append(item)
    assert filter_vals is not None and filter_opts is not None
    filter_mask = np.logical_and(
        filter_mask, np.in1d(filter_vals, filter_opts)
    )
    data = data[filter_mask, :]
    print("%d obs left after filtering." % data.shape[0])

group_split = cmd_args.grouping.split("//")
if len(group_split) == 1:
    grouping = data.obs[group_split[0]]
elif len(group_split) == 2:
    grouping = Cell_BLAST.data.read_hybrid_path(cmd_args.grouping)
    if cmd_args.filters:
        grouping = grouping[filter_mask]
assert len(grouping) == data.exprs.shape[0]

start_time = time.time()
result = fast_markers(data.exprs, group=grouping, fnames=data.var_names,
                      alternative=cmd_args.alternative, n_jobs=cmd_args.n_jobs)
print("Finished in %.3f secs." % (time.time() - start_time))

if not os.path.exists(cmd_args.output_path):
    os.makedirs(cmd_args.output_path)
for key in result:
    filename = "%s.csv" % str(key).replace("/", "_")  # Safe file name
    result[key].to_csv(os.path.join(cmd_args.output_path, filename))

print("Done!")
