import os

configfile: "feature_selection_config.json"

rule feature_selection:
    input:
        f"../Results/feature_selection.{config['plot_ext']}"
    output:
        "../Results/.feature_selection_timestamp"
    threads: 1
    shell:
        "touch {output}"

rule feature_selection_plot:
    input:
        data="../Results/feature_selection.csv",
        script="feature_selection_plot.R"
    output:
        map=f"../Results/feature_selection.{config['plot_ext']}"
    threads: 1
    script:
        "feature_selection_plot.R"

rule feature_selection_summary:
    input:
        [f"{path}/metrics.json" for path in [
            f"../Results/Cell_BLAST/{dataset}/genes_{genes}/seed_{seed}"
            for dataset in config["dataset"]
            for genes in config["genes"]
            for seed in range(config["seed"])
        ] if not os.path.exists(f"{path}/.blacklist")]
    output:
        "../Results/feature_selection.csv"
    params:
        pattern=lambda wildcards: "../Results/Cell_BLAST/{dataset}/genes_{genes}/seed_{seed}/metrics.json"
    threads: 1
    script:
        "summary.py"

rule feature_selection_metrics:
    input:
        data="../Datasets/data/{dataset}/data.h5",
        genes="../Results/Cell_BLAST/{dataset}/genes_{genes}/genes.txt",
        result="../Results/Cell_BLAST/{dataset}/genes_{genes}/seed_{seed}/result.h5"
    output:
        "../Results/Cell_BLAST/{dataset,[^+]+}/genes_{genes}/seed_{seed}/metrics.json"
    threads: 1
    script:
        "feature_selection_metrics.py"

rule feature_selection_prepare:
    input:
        "../Datasets/data/{dataset}/data.h5"
    output:
        "../Results/Cell_BLAST/{dataset}/genes_{genes}/genes.txt"
    log:
        "../Results/Cell_BLAST/{dataset}/genes_{genes}/log.txt"
    params:
        x_low=lambda wildcards: config["genes"][wildcards.genes]["x_low"],
        x_high=lambda wildcards: config["genes"][wildcards.genes]["x_high"],
        y_low=lambda wildcards: config["genes"][wildcards.genes]["y_low"],
        y_high=lambda wildcards: config["genes"][wildcards.genes]["y_high"]
    threads: 1
    shell:
        "python -u feature_selection.py -d {input} -o {output} "
        "--x-low {params.x_low} --x-high {params.x_high} "
        "--y-low {params.y_low} --y-high {params.y_high} "
        "> {log} 2>&1"

rule dimension_reduction_cb:
    input:
        data="../Datasets/data/{dataset}/data.h5",
        genes="../Results/Cell_BLAST/{dataset}/genes_{genes}/genes.txt"
    output:
        "../Results/Cell_BLAST/{dataset,[^+]+}/genes_{genes}/seed_{seed}/result.h5"
    log:
        "../Results/Cell_BLAST/{dataset}/genes_{genes}/seed_{seed}/log.txt"
    params:
        blacklist="../Results/Cell_BLAST/{dataset}/genes_{genes}/seed_{seed}/.blacklist"
    threads: 4
    resources:
        gpu=1
    shell:
        "timeout {config[timeout]} python -u run_DIRECTi.py -i {input.data} -o {output} -g {input.genes} "
        "--prob-module NB -l {config[dimensionality]} -c {config[n_cluster]} "
        "--no-normalize -s {wildcards.seed} --clean {config[label]} > {log} 2>&1 || touch {params.blacklist}"
