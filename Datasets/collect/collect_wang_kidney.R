#! /usr/bin/env Rscript
# by weil
# Nov 03, 2018
# 09:10 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
file_path <- "../download/Wang/"
samples <- dir(file_path)
samples <- samples[grep(".txt", samples)]

gene_name <- read.delim(file.path(file_path, samples[1]))[,1]
expr_mat <- data.frame(row.names = gene_name)

for (sample in samples){
  expr_sample <- read.delim(file.path(file_path, sample))
  rownames(expr_sample) <- expr_sample[,1]
  expr_sample[ , 1] <- NULL
  expr_mat <- cbind(expr_mat, expr_sample)
}

#metadata
meta_df <- read.xlsx(file.path(file_path, "meta_data.xlsx"))
meta_df <- meta_df[2:dim(meta_df)[1], c(1, 7:9)]
colnames(meta_df) <- c("cell_id" ,"cell_type1", "age", "donor")
rownames(meta_df) <- meta_df[,"cell_id"]
meta_df[,"cell_id"] <- NULL
meta_df$organism = "Homo sapiens"
meta_df$organ = "Kidney"
meta_df$lifestage = "fetal"
meta_df$platform = "modified STRT-seq"
meta_df$dataset_name = "Wang_kidney"

#clean cell type
mask <- !grepl("ER.C", meta_df$cell_type1)
meta_df <- meta_df[mask, ]

cell_id <- intersect(colnames(expr_mat), rownames(meta_df))
expr_mat <- expr_mat[, cell_id]
meta_df <- meta_df[cell_id, ]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/kidney_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Wang_kidney", as.matrix(expr_mat), meta_df, datasets_meta, cell_ontology)
message("Done!")
