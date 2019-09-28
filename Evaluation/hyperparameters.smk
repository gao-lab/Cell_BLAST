configfile: "hyperparameters_config.json"

data_pattern = "../Datasets/data/{dataset}/data.h5"
result_pattern_prefix = (
    "../Results/Cell_BLAST/{dataset}/"
    "dim_{dimensionality}_h{hidden_layer}-d{depth}-c{cluster}-l{lambda_prior}-p{prob_module}/"
    "seed_{seed}/"
)
result_pattern = result_pattern_prefix + "result.h5"
metrics_pattern = result_pattern_prefix + "metrics.json"
log_pattern = result_pattern_prefix + "log.txt"

rule hyperparameters:
    input:
        "../Results/hyperparameters_map.{ext}".format(ext=config["plot_ext"])
    output:
        "../Results/.hyperparameters_timestamp"
    threads: 1
    shell:
        "touch {output}"

rule hyperparameters_plot:
    input:
        data="../Results/hyperparameters.csv",
        script="hyperparameters_plot.R"
    output:
        map="../Results/hyperparameters_map.{ext}".format(ext=config["plot_ext"])
    threads: 1
    script:
        "hyperparameters_plot.R"

rule hyperparameters_summary:
    input:
        expand(
            metrics_pattern,
            dimensionality=config["dimensionality"]["pool" if facet == "dimensionality" else "default"],
            hidden_layer=config["hidden_layer"]["pool" if facet == "hidden_layer" else "default"],
            depth=config["depth"]["pool" if facet == "depth" else "default"],
            cluster=config["cluster"]["pool" if facet == "cluster" else "default"],
            lambda_prior=config["lambda_prior"]["pool" if facet == "lambda_prior" else "default"],
            prob_module=config["prob_module"]["pool" if facet == "prob_module" else "default"],
            dataset=config["dataset"],
            seed=range(config["seed"])
        ) for facet in (
            "dimensionality", "hidden_layer", "depth",
            "cluster", "lambda_prior", "prob_module"
        )
    output:
        "../Results/hyperparameters.csv"
    params:
        pattern=lambda wildcards: metrics_pattern
    threads: 1
    script:
        "summary.py"

rule hyperparameters_metrics:
    input:
        data=data_pattern,
        result=result_pattern
    output:
        metrics_pattern
    threads: 1
    script:
        "hyperparameters_metrics.py"

rule hyperparameters_cb:
    input:
        data_pattern
    output:
        result_pattern
    log:
        log_pattern
    threads: 4
    resources:
        gpu=1
    shell:
        "timeout {config[timeout]} python -u run_DIRECTi.py -i {input} -o {output} -g {config[genes]} "
        "--prob-module {wildcards.prob_module} -l {wildcards.dimensionality} -c {wildcards.cluster} "
        "--h-dim {wildcards.hidden_layer} --depth {wildcards.depth} --lambda-prior-reg {wildcards.lambda_prior} "
        "--no-normalize -s {wildcards.seed} --clean {config[label]} > {log} 2>&1"