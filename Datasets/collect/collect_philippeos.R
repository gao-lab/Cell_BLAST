#! /usr/bin/env Rscript
# by weil
# Jan 17, 2019
# 08:59 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat1 <- read.csv("../download/Philippeos/GSE109822_CD3145.csv.gz", row.names = 1)
expr_mat2 <- read.csv("../download/Philippeos/GSE109822_CD90.csv.gz", row.names = 1)

#gene_meta
gene_meta <- expr_mat1[,1, drop = FALSE]

#merge two datasets
expr_mat1$length <- NULL
expr_mat2$length <- NULL
colnames(expr_mat1) <- paste0("3145_", colnames(expr_mat1))
colnames(expr_mat2) <- paste0("90_", colnames(expr_mat2))
expr_mat <- cbind(expr_mat1, expr_mat2)

#meta_df
meta_df <- data.frame(row.names = colnames(expr_mat))
meta_df$cell_type1 = "fibroblast of dermis"
meta_df$region = "dermis"

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/skin_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Philippeos", as.matrix(expr_mat), meta_df, datasets_meta,
                  cell_ontology, gene_meta = gene_meta, x_low = 1, y_low = 1.5)
message("Done!")
