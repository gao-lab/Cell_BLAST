#! /usr/bin/env Rscript
# by weil
# Aug 28, 2018
# 04:18 PM

suppressPackageStartupMessages({
    library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.table("../download/Jose/20180822_PolypAll_cleaned_rawdata.txt", header = TRUE) #, stringsAsFactors = FALSE)
rownames(expr_mat) <- expr_mat[,1]
expr_mat <- expr_mat[,-1]
expr_mat <- as.matrix(expr_mat)

meta_df <- read.table("../download/Jose/20180822_PolypAll_cleaned_metadata.txt", header = TRUE)
rownames(meta_df) <- meta_df[,1]
meta_df[,1] <- NULL
meta_df <- meta_df[ ,c(3:5, 7)]
colnames(meta_df) <- c("donor", "disease", "polyp_disease", "cell_type1")
meta_df$region = "epithelial cells"

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/nasal_cavity_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Ordovas-Montanes", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
