#! /usr/bin/env Rscript
# by weil
# Jan 16, 2019
# 02:10 PM

suppressPackageStartupMessages({
  library(dplyr)
  library(openxlsx)
  library(reshape2)
  library(Matrix)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
#10x
expr_mat_10x <- read.table("../download/Vento-Tormo/raw_data_10x.txt", 
                          sep = "\t", comment.char = "", row.names = 1, header = TRUE)
gene_name_id <- rownames(expr_mat_10x)

#gene_meta
gene_meta <- data.frame(gene_name_id = gene_name_id,
                        gene_name = unlist(lapply(strsplit(gene_name_id, "_EN"), function(x) x[1])),
                        gene_id = unlist(lapply(strsplit(gene_name_id, "_EN"), function(x) x[2])),
                        stringsAsFactors = FALSE)
gene_meta <- gene_meta[!duplicated(gene_meta$gene_name), ]
rownames(gene_meta) <- gene_meta$gene_name
gene_meta$gene_name <- NULL

expr_mat_10x <- expr_mat_10x[gene_meta$gene_name_id, ]
rownames(expr_mat_10x) <- rownames(gene_meta)

#meta_df_10x
meta_df_10x <- read.delim("../download/Vento-Tormo/E-MTAB-6701_arrayexpress_10x_meta.txt", 
                          row.names = 1,
                          stringsAsFactors = FALSE)
rownames(meta_df_10x) <- meta_df_10x$Cell
meta_df_10x$Cell <- NULL
colnames(meta_df_10x) <- c("cluster", "donor", "region", "cell_type1")
meta_df_10x$sample <- unlist(lapply(strsplit(rownames(meta_df_10x), "_"), function(x) x[1]))

# assign cell ontology
cell_ontology <- read.csv("../cell_ontology/placenta_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Vento-Tormo_10x", as.matrix(expr_mat_10x), meta_df_10x, 
                  datasets_meta, cell_ontology, gene_meta = gene_meta, grouping = "donor", y_low = 0.5)
message("Done!")



#Smart-seq2
expr_mat_ss2 <- read.table("../download/Vento-Tormo/raw_data_ss2.txt",
                           sep = "\t", comment.char = "", header = TRUE, row.names = 1)
colnames(expr_mat_ss2) <- gsub("\\.", "_", colnames(expr_mat_ss2))
colnames(expr_mat_ss2) <- substring(colnames(expr_mat_ss2), 2, 20)

expr_mat_ss2 <- expr_mat_ss2[gene_meta$gene_name_id, ]
rownames(expr_mat_ss2) <- rownames(gene_meta)

#meta_df_ss2
meta_df_ss2 <- read.table("../download/Vento-Tormo/E-MTAB-6678_arrayexpress_ss2_meta.txt",
                          sep = "\t", header = TRUE, comment.char = "", 
                          row.names = 1, stringsAsFactors = FALSE)
meta_df_ss2$Cell <- gsub("#", "_", meta_df_ss2$Cell)
rownames(meta_df_ss2) <- meta_df_ss2$Cell
meta_df_ss2$Cell <- NULL
colnames(meta_df_ss2) <- c("cluster", "donor", "region", "cell_type1")

expr_mat_ss2 <- expr_mat_ss2[, rownames(meta_df_ss2)]

construct_dataset("../data/Vento-Tormo_Smart-seq2", as.matrix(expr_mat_ss2), meta_df_ss2, 
                  datasets_meta, cell_ontology, gene_meta = gene_meta, grouping = "donor")
message("Done!")

