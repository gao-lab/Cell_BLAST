#! /usr/bin/env Rscript
# by weil
# Nov 03, 2018
# 06:27 PM

#suppressPackageStartupMessages({
#})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.delim("../download/Guo/GSE112013_Combined_UMI_table.txt")
rownames(expr_mat) <- expr_mat[,1]
expr_mat[,1] <- NULL
expr_mat <- as.matrix(expr_mat)

#metadata
meta <- read.delim("../download/Guo/Guo_metadata.txt", header = F, sep = " ")
meta <- meta[, 2:dim(meta)[2]]
meta_df <- meta[, 1:4]
colnames(meta_df) <- c("cell_id", "ngenes", "nUMI", "cluster_id")
meta_df <- meta_df[2:dim(meta_df)[1], ]
for (i in 2 : (dim(meta)[2]/4)){
  meta_data <-  meta[, (4*i -3):(4*i)]
  colnames(meta_data) <- colnames(meta_df)
  meta_df <- rbind(meta_df, meta_data)
}
meta_df <- meta_df[,c(1,4)]
meta_df <- na.omit(meta_df)
meta_df[,1] <- gsub("-", ".", meta_df[,1])
meta_df$donor <- substring(meta_df$cell_id, 1, 6)
#donor and cluster annotation are from the article
donor_annotation <- data.frame(donor = c("Donor1", "Donor2", "Donor3"),
                               age = c(17,24,35))
meta_df <- merge(meta_df, donor_annotation, by = "donor")
cluster_annotation <- read.csv("../download/Guo/cluster_annotation.csv",header = T)
meta_df <- merge(meta_df, cluster_annotation, by = "cluster_id")
rownames(meta_df) <- meta_df$cell_id
meta_df$cell_id <- NULL
meta_df <- meta_df[colnames(expr_mat), ]
meta_df$cell_type1 <- meta_df$cell_type2
meta_df$cell_type2 <- NULL

cell_ontology <- read.csv("../cell_ontology/testis_cell_ontology.csv")

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Guo", expr_mat, meta_df, datasets_meta, cell_ontology, grouping = "donor")
message("Done!")
