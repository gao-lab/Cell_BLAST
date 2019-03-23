#! /usr/bin/env Rscript
# by weil
# Jan 15, 2019
# 01:28 PM

#Smart-seq2 cells without fitering

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
#Smart-seq2
expr_mat1 <- read.table("../download/Dahlin/GSE106973_HTSeq_counts.txt",
                       sep = "\t", header = TRUE, row.names = 1)
#gene_meta
gene_meta1 <- data.frame(gene_id = rownames(expr_mat1), stringsAsFactors = FALSE)
mouse_94 <- read.csv("../../../GENOME/mapping/mouse_94.txt", header = TRUE, stringsAsFactors = FALSE)
gene_meta1 <- merge(gene_meta1, mouse_94, by.x = "gene_id", by.y = "Gene.stable.ID", all = FALSE)
gene_meta1 <- gene_meta1[!duplicated(gene_meta1$Gene.name), ]
expr_mat1 <- expr_mat1[gene_meta1$gene_id, ]
rownames(expr_mat1) <- gene_meta1$Gene.name
rownames(gene_meta1) <- gene_meta1$Gene.name
gene_meta1$Gene.name <- NULL

#cell_meta
meta_df1 <- data.frame(row.names = colnames(expr_mat1))
meta_df1$cell_type1 <- unlist(lapply(strsplit(rownames(meta_df1), "_"), function(x) x[1]))
meta_df1$organism = "Mus musculus"
meta_df1$organ = "Bone marrow"
meta_df1$platform = "Smart-seq2"
meta_df1$dataset_name = "Dahlin_Smart-seq2"

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/bone_marrow_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Dahlin_Smart-seq2", as.matrix(expr_mat1), meta_df1,
                  datasets_meta, cell_ontology, gene_meta = gene_meta1)
message("Done!")

#10x data after quality control
#WT
expr_mat2 <- read.table("../download/Dahlin/GSM2877127_SIGAB1_counts.txt.gz",
                        sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
expr_mat3 <- read.table("../download/Dahlin/GSM2877128_SIGAC1_counts.txt.gz",
                        sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
expr_mat4 <- read.table("../download/Dahlin/GSM2877129_SIGAD1_counts.txt.gz",
                        sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
expr_mat5 <- read.table("../download/Dahlin/GSM2877130_SIGAF1_counts.txt.gz",
                        sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
expr_mat6 <- read.table("../download/Dahlin/GSM2877131_SIGAG1_counts.txt.gz",
                        sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
expr_mat7 <- read.table("../download/Dahlin/GSM2877132_SIGAH1_counts.txt.gz",
                        sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
colnames(expr_mat2) <- paste0(colnames(expr_mat2), "_SIGAB1")
colnames(expr_mat3) <- paste0(colnames(expr_mat3), "_SIGAC1")
colnames(expr_mat4) <- paste0(colnames(expr_mat4), "_SIGAD1")
colnames(expr_mat5) <- paste0(colnames(expr_mat5), "_SIGAF1")
colnames(expr_mat6) <- paste0(colnames(expr_mat6), "_SIGAG1")
colnames(expr_mat7) <- paste0(colnames(expr_mat7), "_SIGAH1")
expr_mat_10x <- cbind(expr_mat2, expr_mat3, expr_mat4, expr_mat5, expr_mat6, expr_mat7)
# colnames(expr_mat_10x) <- gsub( "\\.", "-", colnames(expr_mat_10x))

#qc
pass_qc_wt <- read.table("../download/Dahlin/pass_qc_wt.txt", stringsAsFactors = FALSE)[, 1]
expr_mat_10x <- expr_mat_10x[ , pass_qc_wt]

#gene_meta
gene_meta_10x <- data.frame(gene_id = rownames(expr_mat_10x), stringsAsFactors = FALSE)
mouse_94 <- read.csv("../../../GENOME/mapping/mouse_94.txt", header = TRUE, stringsAsFactors = FALSE)
gene_meta_10x <- merge(gene_meta_10x, mouse_94, by.x = "gene_id", by.y = "Gene.stable.ID", all = FALSE)
gene_meta_10x <- gene_meta_10x[!duplicated(gene_meta_10x$Gene.name), ]
expr_mat_10x <- expr_mat_10x[gene_meta_10x$gene_id, ]
rownames(expr_mat_10x) <- gene_meta_10x$Gene.name
rownames(gene_meta_10x) <- gene_meta_10x$Gene.name
gene_meta_10x$Gene.name <- NULL

#cell_meta
meta_df_10x <- data.frame(row.names = colnames(expr_mat_10x))
###meta_df_10x$sample
meta_df_10x$cell_type1 = "HSPC"
meta_df_10x$organism = "Mus musculus"
meta_df_10x$organ = "Bone marrow"
meta_df_10x$platform = "10x"
meta_df_10x$dataset_name = "Dahlin_10x"

construct_dataset("../data/Dahlin_10x", as.matrix(expr_mat_10x), meta_df_10x,
                  datasets_meta, cell_ontology, gene_meta = gene_meta_10x, y_low = 0.1)
message("Done!")

#10x c-Kit mutant data after quality control
expr_mat8 <- read.table("../download/Dahlin/GSM2877133_SIGAG8_counts.txt.gz",
                        sep = "\t", header = TRUE, row.names = 1)
expr_mat9 <- read.table("../download/Dahlin/GSM2877134_SIGAH8_counts.txt.gz",
                        sep = "\t", header = TRUE, row.names = 1)
colnames(expr_mat8) <- paste0(colnames(expr_mat8), "_SIGAG8")
colnames(expr_mat9) <- paste0(colnames(expr_mat9), "_SIGAH8")
expr_mat_mutant <- cbind(expr_mat8, expr_mat9)
colnames(expr_mat_mutant) <- gsub( "\\.", "-", colnames(expr_mat_mutant))

#qc
pass_qc_w41 <- read.table("../download/Dahlin/pass_qc_w41.txt", stringsAsFactors = FALSE)[, 1]
expr_mat_mutant <- expr_mat_mutant[ , pass_qc_w41]

#gene_meta
gene_meta_mutant <- data.frame(gene_id = rownames(expr_mat_mutant), stringsAsFactors = FALSE)
gene_meta_mutant <- merge(gene_meta_mutant, mouse_94, by.x = "gene_id", by.y = "Gene.stable.ID", all = FALSE)
gene_meta_mutant <- gene_meta_mutant[!duplicated(gene_meta_mutant$Gene.name), ]
expr_mat_mutant <- expr_mat_mutant[gene_meta_mutant$gene_id, ]
rownames(expr_mat_mutant) <- gene_meta_mutant$Gene.name
rownames(gene_meta_mutant) <- gene_meta_mutant$Gene.name
gene_meta_mutant$Gene.name <- NULL

#cell_meta
meta_df_mutant <- data.frame(row.names = colnames(expr_mat_mutant))
meta_df_mutant$cell_type1 = "HSPC"
meta_df_mutant$organism = "Mus musculus"
meta_df_mutant$organ = "Bone marrow"
meta_df_mutant$platform = "10x"
meta_df_mutant$dataset_name = "Dahlin_mutant"

construct_dataset("../data/Dahlin_mutant", as.matrix(expr_mat_mutant), meta_df_mutant,
                  datasets_meta, cell_ontology, gene_meta = gene_meta_mutant, y_low = 0.1)
message("Done!")
