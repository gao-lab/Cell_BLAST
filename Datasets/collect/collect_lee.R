#! /usr/bin/env Rscript
# by weil
# Jan 1, 2018
# 09:16 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.csv("../download/Lee/GSE101334_postQC_scaled_count_mat.csv.gz",
                     row.names = 1)
expr_mat <- as.matrix(expr_mat)

meta_df <- data.frame(row.names = colnames(expr_mat))
meta_df$cell_type1 = "mesenchyme"
meta_df$region = "mesenchyme"
meta_df$lifestage = "7-10 week"

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/lung_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Lee", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
