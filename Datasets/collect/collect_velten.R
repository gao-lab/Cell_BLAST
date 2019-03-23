#! /usr/bin/env Rscript
# by weil
# Jan 12, 2019
# 08:39 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat1 <- read.csv("../download/Velten/GSE75478_transcriptomics_raw_filtered_I1.csv.gz",
                     header = TRUE, row.names = 1)
expr_mat1 <- expr_mat1[!grepl("\\?", rownames(expr_mat1)), ]
gene_name1 <- unlist(lapply(strsplit(rownames(expr_mat1), " "), function(x) x[1]))
expr_mat1 <- expr_mat1[!duplicated(gene_name1), ]
gene_name1 <- gene_name1[!duplicated(gene_name1)]
rownames(expr_mat1) <- gene_name1

meta_df1 <- data.frame(row.names = colnames(expr_mat1))
meta_df1$cell_type1 = "HSPC"
meta_df1$donor = "I1"
meta_df1$age = 25
meta_df1$gender = "male"

expr_mat2 <- read.csv("../download/Velten/GSE75478_transcriptomics_raw_filtered_I2.csv.gz",
                      header = TRUE, row.names = 1)
expr_mat2 <- expr_mat2[!grepl("\\?", rownames(expr_mat2)), ]
gene_name2 <- unlist(lapply(strsplit(rownames(expr_mat2), " "), function(x) x[1]))
expr_mat2 <- expr_mat2[!duplicated(gene_name2), ]
gene_name2 <- gene_name2[!duplicated(gene_name2)]
rownames(expr_mat2) <- gene_name2

meta_df2 <- data.frame(row.names = colnames(expr_mat2))
meta_df2$cell_type1 = "HSPC"
meta_df2$donor = "I2"
meta_df2$age = 29
meta_df2$gender = "female"

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/bone_marrow_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#clean ERCC
gene_mask1 <- !grepl("ERCC-", rownames(expr_mat1))
expr_mat1 <- expr_mat1[gene_mask1, ]
gene_mask2 <- !grepl("ERCC-", rownames(expr_mat2))
expr_mat2 <- expr_mat2[gene_mask2, ]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Velten_Smart-seq2", as.matrix(expr_mat1), meta_df1, datasets_meta, cell_ontology)
construct_dataset("../data/Velten_QUARTZ-seq", as.matrix(expr_mat2), meta_df2, datasets_meta, cell_ontology, y_low=1.5)
message("Done!")
