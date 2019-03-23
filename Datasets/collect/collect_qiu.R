#! /usr/bin/env Rscript
# by weil
# Oct 21, 2018
# 06：00 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_data <- read.delim("../download/Xu_Cheng-ran/GSE87375_Read_Count.txt")
expr_data <- expr_data[!duplicated(expr_data[,2]), ]#remove duplicated genes
expr_data <- expr_data[expr_data$Symbol != "", ]#remove genes without gene_name
gene_name <- expr_data[,2]
gene_id <- expr_data[,1]
gene_meta <- data.frame(row.names = gene_name, gene_id = gene_id)

gene_length <- expr_data[,3]
read_count <- expr_data[,4:dim(expr_data)[2]]
cell_id <- colnames(read_count)
rpkm <- data.frame(row.names = gene_name)
for (i in 1:dim(read_count)[2]){
  total_count <- sum(read_count[,i])/1e6
  rpkm[,i] <- 1e3 * read_count[,i]/(total_count * gene_length)
}
colnames(rpkm) <- cell_id

meta_df <- openxlsx::read.xlsx("../download/Xu_Cheng-ran/meta.xlsx")
meta_df <- meta_df[,c(1:3,5:6)]
rownames(meta_df) <- meta_df[,1]
meta_df[,1] <- NULL
colnames(meta_df) <- c("cell_type1", "free_annotation", "batch", "cell_cycle_state")
#clean cell type
mask <- (meta_df$cell_type1 != "/")
meta_df <- meta_df[mask, ]
rpkm <- rpkm[,mask]
meta_df$cell_type1[meta_df$cell_type1 == "α-cell"] <- "alpha"
meta_df$cell_type1[meta_df$cell_type1 == "β-cell"] <- "beta"

#clean ERCC
rpkm <- rpkm[!grepl("ERCC-", rownames(rpkm)),]
gene_meta <- gene_meta[rownames(rpkm), ,drop = FALSE]

cell_ontology <- read.csv("../cell_ontology/pancreas_cell_ontology.csv")

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Qiu", as.matrix(rpkm), meta_df, datasets_meta = datasets_meta, 
                  cell_ontology = cell_ontology, gene_meta = gene_meta)
message("Done!")
