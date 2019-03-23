#! /usr/bin/env Rscript
# by caozj
# Dec 6, 2018
# 4:08:48 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.table(
    "../download/Montoro/GSE103354_Trachea_fullLength_TPM.txt.gz",
    header = TRUE
)
cell_name_split <- strsplit(colnames(expr_mat), "_")
meta_df <- data.frame(
    donor = sapply(cell_name_split, function(x) x[2]),  # 3 WT
    region = sapply(cell_name_split, function(x) x[3]),  # prox / dist
    cell_type1 = sapply(cell_name_split, function(x) x[4]),  # 6 cell types
    row.names = colnames(expr_mat)
)

cell_ontology <- read.csv("../cell_ontology/trachea_cell_ontology.csv")
# datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)

construct_dataset("../data/Montoro_Smart-seq2", as.matrix(expr_mat), meta_df, 
                  datasets_meta = datasets_meta, cell_ontology = cell_ontology)
message("Done!")
