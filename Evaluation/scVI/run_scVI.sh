#!/bin/bash

seeds="$(seq 0 15)"
device_rpl="{d} 1 @pool = ($(cat ../devices.txt)); \$_ = \$pool[(\$job->seq() - 1) % (\$#pool + 1)]"
n_jobs=8

if [ ${benchmark} = "semi_supervision" ]; then
    seeds="$(seq 0 1)"
else
    seeds="$(seq 0 15)"
fi

if [ ! -z ${clean} ]; then clean="--clean ${clean}"; fi

parallel -v --dryrun -j1 -k \
    --rpl "{sup} 3 \$_ = ('${supervision}' ne '' && \$_ ne '') ? (('${label_priority}' eq '') ? '_sup_' . \$_ : '_SUP_' . \$_) : ''" \
    --rpl "{:batch} 1 \$rmbatch = (\$_ =~ m/\\+/); \$_ = \$rmbatch ? '-b ${batch}' : '--no-normalize'" \
    --rpl "{:supervision} 1 \$_ = ('${supervision}' ne '') ? '--supervision ${supervision}' : ''" \
    --rpl "{:label_priority} 1 \$_ = ('${supervision}' ne '' && '${label_priority}' ne '') ? '--label-priority ${label_priority}' : ''" \
    --rpl "{:label_fraction} 3 \$_ = ('${supervision}' ne '' && \$_ ne '') ? '--label-fraction ' . \$_ : ''" \
test -e ../../Results/scVI/{1}/${genes}/dim_{4}{sup}/trial_{2}/result.h5 '||' \
'(' \
    mkdir -p ../../Results/scVI/{1}/${genes}/dim_{4}{sup}/trial_{2} '&&' \
    timeout ${timeout} python -u ./run_scVI.py \
        -i ../../Datasets/data/{1}/data.h5 -g ${genes} \
        -o ../../Results/scVI/{1}/${genes}/dim_{4}{sup}/trial_{2}/ \
        --n-latent {4} {:supervision} {:label_priority} {:label_fraction} {:batch} \
        -s {2} -d {device} ${clean} '>' \
    ../../Results/scVI/{1}/${genes}/dim_{4}{sup}/trial_{2}/output.txt '2>&1' \
')' \
::: ${datasets} ::: ${seeds} ::: ${label_fractions} ::: ${dims} | \
tr -d "'" | tr -s " " | sort | uniq | \
parallel -j1 -k --rpl "${device_rpl}" echo {} '|' sed 's/{device}/{d}/g' | \
parallel -v -j${n_jobs} $1 {}

if [ $? -eq 124 ]; then
    true
fi  # timeout exit code