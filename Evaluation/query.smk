import os

configfile: "query_config.json"

rule query:
    input:
        expand(
            "../Results/query_{item}.{ext}",
            item=["roc", "pnsum", "pnopt", "mba"],
            ext=config["plot_ext"]
        ),
        "../Results/query_cell_type_specific.{ext}".format(
            ext=config["plot_ext"]
        )
    output:
        "../Results/.query_timestamp"
    threads: 1
    shell:
        "touch {output}"

rule query_ct_plot:
    input:
        data="../Results/query_cell_type_specific.xlsx",
        script="query_ct_plot.R"
    output:
        "../Results/query_cell_type_specific.{ext}".format(
            ext=config["plot_ext"]
        )
    threads: 1
    script:
        "query_ct_plot.R"

rule query_ct_summary:
    input:
        data=expand(
            "../Results/{method}/{group}/seed_{seed}/cell_type_specific.xlsx",
            method=config["method"],
            group=config["dataset_group"],
            seed=range(config["seed"])
        ),
        selected="../Results/query_selected.csv",
        script="query_ct_summary.py"
    output:
        "../Results/query_cell_type_specific.xlsx"
    params:
        pattern=lambda wildcards: "../Results/{method}/{group}/seed_{seed}/cell_type_specific.xlsx"
    threads: 1
    script:
        "query_ct_summary.py"

rule query_plot:
    input:
        data="../Results/query.csv",
        palette="palette_method.json",
        script="query_plot.R"
    output:
        roc="../Results/query_roc.{ext}".format(ext=config["plot_ext"]),
        auc="../Results/query_auc.{ext}".format(ext=config["plot_ext"]),
        pnsum="../Results/query_pnsum.{ext}".format(ext=config["plot_ext"]),
        pnopt="../Results/query_pnopt.{ext}".format(ext=config["plot_ext"]),
        mba="../Results/query_mba.{ext}".format(ext=config["plot_ext"]),
        selected="../Results/query_selected.csv"
    threads: 1
    script:
        "query_plot.R"

rule query_summary:
    input:
        expand(
            "../Results/{method}/{group}/seed_{seed}/metrics.json",
            method=config["method"],
            group=config["dataset_group"],
            seed=range(config["seed"])
        )
    output:
        "../Results/query.csv"
    params:
        pattern=lambda wildcards: "../Results/{method}/{group}/seed_{seed}/metrics.json"
    threads: 1
    script:
        "summary.py"

rule query_metrics:
    input:
        ref=lambda wildcards: expand(
            "../Datasets/data/{ref}/data.h5",
            ref=config[wildcards.group]["ref"]
        ),
        true=lambda wildcards: expand(
            "../Datasets/data/{query}/data.h5",
            query=config[wildcards.group]["pos"] + config[wildcards.group]["neg"]
        ),
        pred=lambda wildcards: expand(
            "../Results/{method}/{group}/seed_{seed}/{query}/result.h5",
            method=wildcards.method,
            group=wildcards.group,
            seed=wildcards.seed,
            query=config[wildcards.group]["pos"] + config[wildcards.group]["neg"]
        )
    output:
        "../Results/{method}/{group}/seed_{seed}/metrics.json",
        "../Results/{method}/{group}/seed_{seed}/cell_type_specific.xlsx"
    params:
        expect=lambda wildcards: "../Datasets/expected_predictions/{group}.csv".format(group=wildcards.group)
    threads: 1
    script:
        "query_metrics.py"

include: "dimension_reduction_worker.smk"
include: "batch_removal_worker.smk"
include: "query_worker.smk"