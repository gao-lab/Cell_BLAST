#! /usr/bin/env Rscript
# by weil
# Jan 4, 2018
# 04:16 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat1 <- read.table("../download/Nguyen/GSM3099846_Ind4_Expression_Matrix.txt.gz", 
                        sep = "\t", header = TRUE, row.names = 1)
expr_mat2 <- read.table("../download/Nguyen/GSM3099847_Ind5_Expression_Matrix.txt.gz", 
                        sep = "\t", header = TRUE, row.names = 1)
expr_mat3 <- read.table("../download/Nguyen/GSM3099848_Ind6_Expression_Matrix.txt.gz", 
                        sep = "\t", header = TRUE, row.names = 1)
expr_mat4 <- read.table("../download/Nguyen/GSM3099849_Ind7_Expression_Matrix.txt.gz", 
                        sep = "\t", header = TRUE, row.names = 1)
expr_mat_10x <- cbind(expr_mat1, expr_mat2, expr_mat3, expr_mat4)

#10x metadata
meta_df <- read.table("../download/Nguyen/Kessenbrock_Droplet_Based_Cell_Cluster_IDs.txt", 
                      sep = "\t", header = TRUE)
meta_df$Cell_Name <- gsub("-", "\\.", meta_df$Cell_Name)
colnames(meta_df) <- c("cell_id", "cell_type1")
meta_df$donor <- unlist(lapply(strsplit(meta_df$cell_id, "_"), function(x) x[1]))
meta_df$region = "epithelial cells"
donor_age <- read.xlsx("../download/Nguyen/41467_2018_4334_MOESM4_ESM.xlsx")[, c(1,3,4)]
meta_df <- merge(meta_df, donor_age, by = "donor")
rownames(meta_df) <- meta_df$cell_id
meta_df$cell_id <- NULL
#clean cell type
meta_df <- meta_df[meta_df$cell_type1 != "Unclassified", ]

expr_mat_10x <- expr_mat_10x[, rownames(meta_df)]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/mammary_gland_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Nguyen_10x", as.matrix(expr_mat_10x), meta_df, 
                  datasets_meta, cell_ontology, grouping = "donor")
message("Done!")
