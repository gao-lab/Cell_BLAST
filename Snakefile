subworkflow evaluation:
    workdir:
        "Evaluation"

subworkflow experiments:
    workdir:
        "Notebooks/Experiments"

subworkflow case_study:
    workdir:
        "Notebooks/Case"

rule all:
    input:
        evaluation("../Results/.all_timestamp"),
        experiments(".experiments_timestamp"),
        case_study(".case_study_timestamp")