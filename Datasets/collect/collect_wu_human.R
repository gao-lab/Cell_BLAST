#! /usr/bin/env Rscript
# by weil
# Dec 16, 2018
# 02:46 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.delim("../download/Wu_human/GSE118184_Human_kidney_snRNA.dge.txt.gz")
expr_mat <- as.matrix(expr_mat)

meta_df <- read.xlsx("../download/Wu_human/metadata.xlsx" )
rownames(meta_df) <- meta_df$Cell.Barcode
meta_df$Cell.Barcode <- NULL
colnames(meta_df) <- "cell_type1"

#clean cell type
mask <- !grepl("Undefined", meta_df$cell_type1)
meta_df <- meta_df[mask, ]
expr_mat <- expr_mat[, mask]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/kidney_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Wu_human", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
