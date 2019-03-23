#!/bin/bash

set -e

export benchmark="bias_removal"
export methods="PCA MNN CCA scVI Cell_BLAST"  # PCA for negative control

export datasets="Baron_human+Baron_mouse Baron_human+Muraro+Enge+Segerstolpe+Xin_2016+Lawlor Montoro_10x+Plasschaert Quake_Smart-seq2+Quake_10x"
export clean="cell_ontology_class"
export batch="dataset_name"
export genes="seurat_genes"
export dims="10"
export timeout="2h"
export lambda_rmbatch_regs="0.0 0.001 0.005 0.01 0.05 0.1 1.0"

for method in ${methods}; do
    (cd ${method} && ./run_${method}.sh)
done

./benchmark_bias_removal.py
./benchmark_bias_removal.R
