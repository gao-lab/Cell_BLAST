#! /usr/bin/env Rscript
# by weil
# Jan 3, 2018
# 09:42 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.table("../download/Giraddi/GSE111113_Table_S1_FilterNormal10xExpMatrix.txt.gz",
                       sep = "\t", header = TRUE)

gene_meta <- expr_mat[,1:3]
#drop duplicated genes
gene_mask <- !duplicated(gene_meta[,3])
gene_meta <- gene_meta[gene_mask, ]
rownames(gene_meta) <- gene_meta[,3]
gene_meta[,3] <- NULL

meta_df <- read.table("../download/Giraddi/celrep_5308_mmc6.txt", 
                      sep = "\t", header = TRUE, row.names = 1)
colnames(meta_df) <- c("pseudotime", "cell_type1")
meta_df$lifestage <- unlist(lapply(strsplit(rownames(meta_df), "_"), function(x) x[1]))

#clean cell type
cell_mask <- meta_df$cell_type1 != "N/A" & meta_df$cell_type1 != "indeterminant/stromal"
meta_df <- meta_df[cell_mask, ]

expr_mat <- as.matrix(expr_mat[gene_mask, rownames(meta_df)])
rownames(expr_mat) <- rownames(gene_meta)
expr_mat <- as.matrix(expr_mat)

#clean ERCC
gene_mask_ERCC <- !grepl("ERCC-", rownames(expr_mat))
expr_mat <- expr_mat[gene_mask_ERCC, ]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/mammary_gland_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Giraddi_10x", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")

#C1 platform
expr_mat_C1 <- read.table("../download/Giraddi/celrep_5308_mmc3.txt",
                          sep ="\t", header = TRUE)

gene_meta_C1 <- expr_mat_C1[,1:3]
#drop duplicated genes
gene_mask_C1 <- !duplicated(gene_meta_C1[,3])
gene_meta_C1 <- gene_meta_C1[gene_mask_C1, ]
rownames(gene_meta_C1) <- gene_meta_C1[,3]
gene_meta_C1[,3] <- NULL

expr_mat_C1 <- as.matrix(expr_mat_C1[gene_mask_C1, 4:dim(expr_mat_C1)[2]])
rownames(expr_mat_C1) <- rownames(gene_meta_C1)

meta_df_C1 <- data.frame(row.names = colnames(expr_mat_C1))
meta_df_C1$lifestage <- unlist(lapply(strsplit(rownames(meta_df_C1), "M"), function(x) x[1]))

construct_dataset("../data/Giraddi_C1", expr_mat_C1, meta_df_C1, datasets_meta)
message("Done!")
