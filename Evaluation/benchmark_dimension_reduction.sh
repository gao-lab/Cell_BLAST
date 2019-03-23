#!/bin/bash

set -e

export benchmark="dimension_reduction"
export methods="PCA ZIFA ZINB_WaVE scVI Cell_BLAST"

export datasets="Muraro Adam Guo Plasschaert Baron_human Bach Macosko"
export clean="cell_ontology_class"
export genes="seurat_genes"
export dims="10"
export timeout="2h"

for method in ${methods}; do
    (cd ${method} && ./run_${method}.sh)
done

./benchmark_dimension_reduction.py
./benchmark_dimension_reduction.R
