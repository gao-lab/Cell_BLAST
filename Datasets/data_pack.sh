#!/bin/bash

set -e

function pack {
    args=("$@")
    output="${args[0]}"
    datasets=("${args[@]:1}")
    rm -rf "${output}"
    mkdir -p "${output}/data"
    for dataset in "${datasets[@]}"; do
        mkdir "${output}/data/${dataset}"
        ln -rs "data/${dataset}/data.h5" "${output}/data/${dataset}/data.h5"
    done
    tar czvhf "${output}.tar.gz" -C "${output}" "data"
    rm -rf "${output}"
}


#----------------------------------- Basic -------------------------------------

basic_output="data_pack"
basic_datasets=(
    "Guo" "Muraro" "Xin_2016" "Lawlor" "Segerstolpe" "Enge" "Baron_human"
    "Baron_mouse" "Adam" "Plasschaert" "Montoro_10x" "Macosko" "Bach"
    "Quake_Smart-seq2" "Quake_10x" "Wu_human" "Zheng" "Philippeos" "Park"
    "Giraddi_10x" "Quake_Smart-seq2_Mammary_Gland" "Quake_10x_Mammary_Gland"
    "Quake_10x_Lung" "Quake_Smart-seq2_Lung" "Montoro_10x_noi" "Plasschaert_noi"
    "Tusi" "Velten_Smart-seq2"
)
echo "Packing basic datasets..."
pack "${basic_output}" "${basic_datasets[@]}"


#---------------------------------- Extended -----------------------------------

ext_output="data_pack_ext"
ext_datasets=(
    "1M_neurons" "1M_neurons_half" "Marques"
)
echo "Packing extended datasets..."
pack "${ext_output}" "${ext_datasets[@]}"
