#! /usr/bin/env Rscript
# by weil
# Jan 18, 2019
# 07:53 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat1 <- read.table("../download/Alles/GSM2518777_mel_rep1_dge.txt.gz", 
                        header = TRUE, row.names = 1)
expr_mat2 <- read.table("../download/Alles/GSM2518778_mel_rep2_dge.txt.gz", 
                        header = TRUE, row.names = 1)
expr_mat3 <- read.table("../download/Alles/GSM2518779_mel_rep3_dge.txt.gz", 
                        header = TRUE, row.names = 1)
expr_mat4 <- read.table("../download/Alles/GSM2518780_mel_rep4_dge.txt.gz", 
                        header = TRUE, row.names = 1)
expr_mat5 <- read.table("../download/Alles/GSM2518781_mel_rep5_dge.txt.gz", 
                        header = TRUE, row.names = 1)
expr_mat6 <- read.table("../download/Alles/GSM2518782_mel_rep6_dge.txt.gz", 
                        header = TRUE, row.names = 1)
expr_mat7 <- read.table("../download/Alles/GSM2518783_mel_rep7_dge.txt.gz", 
                        header = TRUE, row.names = 1)
genes <- unique(c(rownames(expr_mat1), rownames(expr_mat2), rownames(expr_mat3), rownames(expr_mat4), 
               rownames(expr_mat5), rownames(expr_mat6), rownames(expr_mat7)))
expr_1 <- expr_mat1[genes, ]
expr_2 <- expr_mat2[genes, ]
expr_3 <- expr_mat3[genes, ]
expr_4 <- expr_mat4[genes, ]
expr_5 <- expr_mat5[genes, ]
expr_6 <- expr_mat6[genes, ]
expr_7 <- expr_mat7[genes, ]
expr_mat <- cbind(expr_1, expr_2, expr_3, expr_4, expr_5, expr_6,expr_7)
rownames(expr_mat) <- genes
expr_mat[is.na(expr_mat)] <- 0

#meta_df
meta_df <- read.table("../download/Alles/GSE89164_clusters_dmel.txt.gz")
colnames(meta_df) <- c("barcode", "cluster")
meta_df$barcode <- unlist(lapply(strsplit(as.character(meta_df$barcode), "_"), function(x) x[1]))
cluster_annotation <- read.csv("../download/Alles/cluster_annotation.csv")
meta_df <- merge(meta_df, cluster_annotation, by = "cluster")
meta_df$lifestage = "embryo"
rownames(meta_df) <- meta_df$barcode
meta_df$barcode <- NULL

#clean cell type
meta_df <- meta_df[meta_df$cell_type1 != "not assigned" & meta_df$cell_type1 != "fat body/hemocyte", ]
expr_mat <- expr_mat[, rownames(meta_df)]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/drosophila_embryo_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names=1)
construct_dataset("../data/Alles", as.matrix(expr_mat), meta_df, datasets_meta, cell_ontology, y_low = 0.1)
message("Done!")
