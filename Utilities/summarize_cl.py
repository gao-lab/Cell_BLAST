#!/usr/bin/env python

import os
import numpy as np
import Cell_BLAST as cb

path = "../Datasets/data"
datasets = os.listdir(path)
for dataset in datasets:
    try:
        cl = cb.data.read_hybrid_path(os.path.join(
            path, dataset, "data.h5//obs/cell_ontology_class"))
        print(dataset + ": ", end="")
        unique_cl, count = np.unique(cl, return_counts=True)
        print(unique_cl.tolist(), end="")
        print(", ", end="")
        print(count.tolist())
    except Exception:
        pass
