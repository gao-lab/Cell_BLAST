#!/bin/bash

host="caozj@gpu1"
exclude="--exclude .git --exclude .snakemake --exclude .ipynb_checkpoints --exclude __pycache__ --exclude *.pyc"

if [ "${1}" = "pull" ]; then
    rsync ${2} -avzhPL ${exclude} \
        --include "/Datasets" \
            --include "/Datasets/data" \
            --include "/Datasets/expected_predictions" \
            --include "/Datasets/ortholog" \
                --include "/Datasets/ortholog/Ensembl" \
                --exclude "/Datasets/ortholog/*" \
            --exclude "/Datasets/*" \
        --include "/Evaluation" \
        --include "/Utilities" \
        --include "/dist" \
        --include "/local" \
            --include "/local/*.tar.gz" \
            --exclude "/local/*" \
        --include "/packrat" \
            --exclude "/packrat/lib*" \
            --exclude "/packrat/envs/seurat_v3/packrat/lib*" \
        --include "/test" \
        --include "/env.yml" \
        --include "/.Rprofile" \
        --exclude "/*" \
        "${host}:~/SC/*" "./"
elif [ "${1}" = "push" ]; then
    rsync ${2} -avzhP "Results/*" "${host}:/rd3/caozj/SC/Results/"
else
    echo "Usage: ./sync.sh (pull | push)"
fi
