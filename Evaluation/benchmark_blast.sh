#!/bin/bash

# set -e

RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

export methods="scmap CellFishing.jl Cell_BLAST"
export dims="10"  # Only for Cell_BLAST
export lambda_rmbatch_regs="0.01"  # Only for Cell_BLAST
export timeout="2h"
export clean="cell_ontology_class"
export annotate="cell_ontology_class"

# Preparation
echo -e "${RED}"
for method in ${methods}; do
    if [ ${method} = "Cell_BLAST" ]; then
        export benchmark="dimension_reduction"
        export genes="seurat_genes"
        export datasets="Montoro_10x Bach Quake_10x_Spleen Quake_10x_Lung"
        (cd ${method} && ./run_${method}.sh)
        export benchmark="bias_removal"
        export batch="dataset_name"
        export datasets="Baron_human+Xin_2016+Lawlor"
        (cd ${method} && ./run_${method}.sh)
    fi
done
echo -e "${NOCOLOR}"


datasets_arr=(
    Baron_human+Xin_2016+Lawlor:Muraro
    Baron_human+Xin_2016+Lawlor:Segerstolpe
    Baron_human+Xin_2016+Lawlor:Enge
    Baron_human+Xin_2016+Lawlor:Young
    Baron_human+Xin_2016+Lawlor:Wu_human
    Baron_human+Xin_2016+Lawlor:Zheng_subsample
    Baron_human+Xin_2016+Lawlor:Philippeos
    Montoro_10x:Plasschaert
    Montoro_10x:Baron_mouse
    Montoro_10x:Park
    Montoro_10x:Bach
    Montoro_10x:Macosko
    Bach:Sun
    Bach:Giraddi_10x
    Bach:Quake_Smart-seq2_Mammary_Gland
    Bach:Quake_10x_Mammary_Gland
    Bach:Baron_mouse
    Bach:Park
    Bach:Plasschaert
    Bach:Macosko
    Quake_10x_Spleen:Quake_Smart-seq2_Spleen
    Quake_10x_Spleen:Baron_mouse
    Quake_10x_Spleen:Park
    Quake_10x_Spleen:Bach
    Quake_10x_Spleen:Plasschaert
    Quake_10x_Lung:Quake_Smart-seq2_Lung
    Quake_10x_Lung:Baron_mouse
    Quake_10x_Lung:Park
    Quake_10x_Lung:Macosko
    Quake_10x_Lung:Bach
)


export benchmark="blast"
export datasets="${datasets_arr[*]}"
export scmap_thresholds="0.0 0.1 0.2 0.3 0.4 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.9 1.0"
export CellFishing_thresholds="0,20,50,70,80,90,100,110,120,130,140,160,180,200"
export Cell_BLAST_thresholds="0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.05 0.1 0.2 0.3 0.4 0.5 1.0"

# BLAST
echo -e "${GREEN}"
for method in ${methods}; do
    if [ ${method} = "Cell_BLAST" ]; then
        export genes="seurat_genes"
        export align="no"
        (cd ${method} && ./run_${method}.sh)
        export align="yes"
        (cd ${method} && ./run_${method}.sh)
    elif [ ${method} = "scmap" ]; then
        export genes="scmap_genes"
        (cd ${method} && ./run_${method}.sh)
    else  # CellFishing.jl
        export genes="cf_genes"
        (cd ${method} && ./run_${method}.sh)
    fi
done
echo -e "${NOCOLOR}"

./benchmark_blast.py

export methods="$methods Cell_BLAST_aligned"
export scmap_default_threshold=0.5
export CellFishing_default_threshold=110
export Cell_BLAST_default_threshold=0.05
export include_align="Quake_10x_Spleen"
./benchmark_blast.R
