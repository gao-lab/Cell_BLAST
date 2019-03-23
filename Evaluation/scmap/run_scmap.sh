#!/bin/bash

if [ -z "${n_jobs}" ]; then n_jobs=4; fi
if [ -z "${seeds}" ]; then seeds="$(seq 0 3)"; fi
if [ ! -z "${clean}" ]; then clean="--clean ${clean}"; fi
if [ ! -z "${subsample_ref}" ]; then
    subsample_suffix="_${subsample_ref}"
    subsample_ref="--subsample-ref ${subsample_ref}"
fi

parallel -j1 echo {1}:{2} ::: ${datasets} ::: ${seeds} | \
parallel -v --colsep ':' $1 -j${n_jobs} \
    test -e ../../Results/scmap/{1}/${genes}/trial_{3}/{2}${subsample_suffix}.h5 '||' \
    '(' \
        mkdir -p ../../Results/scmap/{1}/${genes}/trial_{3} '&&' \
        ./run_scmap.R \
            -r '$(' \
                echo -n {1} '|' parallel -j1 -k -d '+' --rpl \'{i} 1 \$_=\$_\' \
                echo ../../Datasets/data/{i}/data.h5 \
            ')' \
            -n '$(' echo {1} '|' tr \'+\' \' \' ')' \
            -q ../../Datasets/data/{2}/data.h5 \
            -o ../../Results/scmap/{1}/${genes}/trial_{3}/{2}${subsample_suffix}.h5 \
            -g ${genes} --threshold ${scmap_thresholds} -s {3} \
            --cluster-col ${annotate} ${subsample_ref} ${clean} --shuffle-genes '>' \
        ../../Results/scmap/{1}/${genes}/trial_{3}/{2}${subsample_suffix}_output.txt '2>&1' \
    ')'
