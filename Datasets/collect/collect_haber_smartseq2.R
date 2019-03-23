#! /usr/bin/env Rscript
# by weil
# Aug 31, 2018
# 05:16 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- as.matrix(read.table("../download/Haber/SMART-Seq/GSE92332_AtlasFullLength_TPM.txt"))
meta_df <- data.frame(row.names = colnames(expr_mat))
meta_df$cell_type1 <- unlist(lapply(strsplit(rownames(meta_df), "_"), function(x) x[4]))
meta_df$region = "epithelial cells"

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/small_intestine_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Haber_Smart-seq2/", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
