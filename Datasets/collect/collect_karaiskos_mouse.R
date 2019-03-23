#! /usr/bin/env Rscript
# by weil
# Dec 8, 2018
# 03:34 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.delim("../download/Karaiskos/dge_filtered_clusters.txt")
expr_mat <- as.matrix(expr_mat)

meta_df <- data.frame(row.names = colnames(expr_mat))
meta_df$cell_type1 <- as.vector(unlist(lapply(strsplit(colnames(expr_mat), "_"), function(x) x[3])))
meta_df$replicate <- as.vector(unlist(lapply(strsplit(colnames(expr_mat), "_"), function(x) x[2])))
meta_df$organism = "Mus musculus"
meta_df$organ = "Kidney"
meta_df$region = "Glomeruli"
meta_df$platform = "Drop-seq"
meta_df$dataset_name = "Karaiskos_mouse"

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/kidney_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Karaiskos_mouse", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
