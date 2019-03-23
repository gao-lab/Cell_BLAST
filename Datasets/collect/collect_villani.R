#! /usr/bin/env Rscript
# by weil
# Dec 12, 2018
# 03:46 PM

suppressPackageStartupMessages({
    library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.delim("../download/Villani/GSE94820_raw.expMatrix_DCnMono.discovery.set.submission.txt")
expr_mat <- as.matrix(expr_mat)

meta_df <- read.xlsx("../download/Villani/NIHMS910854-supplement-Supplementary_Tables_1-16.xlsx", sheet = "TableS15.DiscoveryCells")
meta_df <- meta_df[2:dim(meta_df)[1], c(2,6)]
colnames(meta_df) <- c("cell_id", "cell_type2")
cell_type <- read.xlsx("../download/Villani/cell_type.xlsx")
meta_df <- merge(meta_df, cell_type, by = "cell_type2")
rownames(meta_df) <- meta_df$cell_id
meta_df$cell_id <- NULL
meta_df$region = "DCs and monocytes"

expr_mat <- expr_mat[, rownames(meta_df)]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/blood_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Villani", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
