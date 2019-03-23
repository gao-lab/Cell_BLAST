#! /usr/bin/env Rscript
# by weil
# Jan 4, 2018
# 09:01 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.table("../download/Sun/GSE110371_All_WT_Gene_expression.txt.gz",
                     header = TRUE, row.names = 1)
expr_mat <- as.matrix(expr_mat)

meta_df <- data.frame(cell_id = colnames(expr_mat))
meta_df$lifestage <- substring(meta_df$cell_id, 1, 1)
meta_df$cell_type1 <- substring(meta_df$cell_id, 2, 2)
meta_df$lifestage <- gsub("V", "virgin", meta_df$lifestage)
meta_df$lifestage <- gsub("P", "pregnant", meta_df$lifestage)
meta_df$cell_type1 <- gsub("B", "basal cell", meta_df$cell_type1)
meta_df$cell_type1 <- gsub("L", "luminal cell", meta_df$cell_type1)
meta_df$organism = "Mus musculus"
meta_df$organ = "Mammary gland"
meta_df$region = "epithelial cells"
meta_df$platform = "C1"
meta_df$dataset_name = "Sun"
rownames(meta_df) <- meta_df$cell_id
meta_df$cell_id <- NULL

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/mammary_gland_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Sun", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
