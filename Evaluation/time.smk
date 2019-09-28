configfile: "time_config.json"

rule time:
    input:
        "../Results/time.pdf"
    output:
        "../Results/.time_timestamp"
    threads: 1
    shell:
        "touch {output}"

rule time_plot:
    input:
        data="../Results/time.csv",
        palette="palette_method.json",
        script="time_plot.R"
    output:
        "../Results/time.pdf"
    script:
        "time_plot.R"

rule time_summary:
    input:
        ["{path}/metrics.json".format(path=item) for item in expand(
            "../Results/{method}/{ref}/size_{size}/seed_{seed}/{query}",
            method=config["method"],
            ref=config["ref"],
            size=config["size"],
            seed=range(config["seed"]),
            query=config["query"]
        ) if not os.path.exists("{path}/.blacklist".format(path=item))]
    output:
        "../Results/time.csv"
    params:
        pattern=lambda wildcards: "../Results/{method}/{ref}/size_{size}/seed_{seed}/{query}/metrics.json"
    threads: 1
    script:
        "summary.py"

rule time_metrics:
    input:
        "../Results/{method}/{ref}/size_{size}/seed_{seed}/{query}/result.h5"
    output:
        "../Results/{method}/{ref}/size_{size}/seed_{seed}/{query}/metrics.json"
    threads: 1
    script:
        "time_metrics.py"

rule time_scmap:
    input:
        ref=lambda wildcards: "../Datasets/data/{ref}/data.h5".format(
            ref=config["scmap"]["alt_ref"][wildcards.ref] \
                if wildcards.ref in config["scmap"]["alt_ref"] else wildcards.ref
        ),  # Full 1M neuron too large to work in R
        query="../Datasets/data/{query}/data.h5"
    output:
        "../Results/scmap/{ref}/size_{size}/seed_{seed}/{query}/result.h5"
    log:
        "../Results/scmap/{ref}/size_{size}/seed_{seed}/{query}/log.txt"
    params:
        blacklist="../Results/scmap/{ref}/size_{size}/seed_{seed}/{query}/.blacklist"
    threads: 1
    shell:
        "timeout {config[timeout]} Rscript run_scmap.R "
        "-r {input.ref} -g {config[genes]} -c {config[label]} -q {input.query} "
        "-o {output} -s {wildcards.seed} --shuffle-genes "
        "--subsample-ref {wildcards.size} > {log} 2>&1 || touch {params.blacklist}"

rule time_cf:
    input:
        ref="../Datasets/data/{ref}/data.h5",
        query="../Datasets/data/{query}/data.h5"
    output:
        "../Results/CellFishing.jl/{ref}/size_{size}/seed_{seed}/{query}/result.h5"
    log:
        "../Results/CellFishing.jl/{ref}/size_{size}/seed_{seed}/{query}/log.txt"
    params:
        blacklist="../Results/CellFishing.jl/{ref}/size_{size}/seed_{seed}/{query}/.blacklist"
    threads: 1
    shell:
        "timeout {config[timeout]} julia run_CellFishing.jl.jl "
        "--annotation={config[label]} --gene={config[genes]} "
        "--subsample-ref={wildcards.size} --seed={wildcards.seed} "
        "{input.ref} {input.query} {output} > {log} 2>&1 || touch {params.blacklist}"

rule time_cb_build:
    input:
        ref="../Datasets/data/{ref}/data.h5",
        models=lambda wildcards: expand(
            "../Results/Cell_BLAST/{ref}/dim_{dimensionality}/seed_{seed}/result.h5",
            ref=wildcards.ref,
            dimensionality=config["Cell_BLAST"]["dimensionality"],
            seed=range(
                int(wildcards.seed) * config["Cell_BLAST"]["n_models"],
                (int(wildcards.seed) + 1) * config["Cell_BLAST"]["n_models"]
            )
        )
    output:
        directory("../Results/Cell_BLAST/{ref}/seed_{seed}/blast")
    log:
        "../Results/Cell_BLAST/{ref}/seed_{seed}/log.txt"
    params:
        models=lambda wildcards, input: [os.path.dirname(item) for item in input.models]
    threads: 1
    shell:
        "timeout {config[timeout]} python -u build_BLAST.py "
        "-r {input.ref} -m {params.models} -o {output} -j {config[Cell_BLAST][n_models]} "
        "-s {wildcards.seed} > {log} 2>&1"

rule time_cb_query:
    input:
        blast="../Results/Cell_BLAST/{ref}/seed_{seed}/blast",
        query="../Datasets/data/{query}/data.h5"
    output:
        "../Results/Cell_BLAST/{ref}/size_{size}/seed_{seed}/{query}/result.h5"
    log:
        "../Results/Cell_BLAST/{ref}/size_{size}/seed_{seed}/{query}/log.txt"
    params:
        blacklist="../Results/Cell_BLAST/{ref}/size_{size}/seed_{seed}/{query}/.blacklist"
    threads: 1
    shell:
        "timeout {config[timeout]} python -u run_BLAST.py "
        "-i {input.blast} -o {output} -q {input.query} -a {config[label]} "
        "-j {config[Cell_BLAST][n_models]} -s {wildcards.seed} "
        "--subsample-ref {wildcards.size} > {log} 2>&1 || touch {params.blacklist}"

rule time_dca_query:
    input:
        ref="../Datasets/data/{ref}/data.h5",
        model="../Results/DCA/{{ref}}/dim_{dimensionality}/seed_{{seed}}/result.h5".format(
            dimensionality=config["DCA"]["dimensionality"]
        ),  # Actually model results
        query="../Datasets/data/{query}/data.h5"
    output:
        "../Results/DCA/{ref}/size_{size}/seed_{seed}/{query}/result.h5"
    log:
        "../Results/DCA/{ref}/size_{size}/seed_{seed}/{query}/log.txt"
    params:
        model=lambda wildcards, input: os.path.dirname(input.model)
    threads: 1
    shell:
        "timeout {config[timeout]} python -u run_DCA_query.py "
        "-m {params.model} -r {input.ref} -q {input.query} -o {output} "
        "-a {config[label]} -s {wildcards.seed} "
        "--subsample-ref {wildcards.size} > {log} 2>&1"

include: "dimension_reduction_worker.smk"