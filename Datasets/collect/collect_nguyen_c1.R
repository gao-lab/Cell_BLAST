#! /usr/bin/env Rscript
# by weil
# Jan 4, 2018
# 04:16 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
file_path <- "../download/Nguyen"
mat_C1 <- dir("../download/Nguyen/", "fpkm")
expr_mat_C1 <- data.frame(X = 1)
for (mat in mat_C1){
  expr_mat_C1 <- merge(expr_mat_C1, 
                       read.table(file.path(file_path, mat), sep = "\t", header = TRUE), 
                       by = 1, all.y = TRUE)
}
rownames(expr_mat_C1) <- expr_mat_C1[,1]
expr_mat_C1[,1] <- NULL

meta_df_C1 <- data.frame(cell_id = colnames(expr_mat_C1))
meta_df_C1$donor <- unlist(lapply(strsplit(colnames(expr_mat_C1), "_"), function(x) x[2]))
meta_df_C1$library <- unlist(lapply(strsplit(colnames(expr_mat_C1), "_"), function(x) x[1]))
meta_df_C1$cell_type1 <- unlist(lapply(strsplit(colnames(expr_mat_C1), "_"), function(x) x[3]))
donor_age <- read.xlsx("../download/Nguyen/41467_2018_4334_MOESM4_ESM.xlsx")[, c(1,3,4)]
meta_df_C1 <- merge(meta_df_C1, donor_age, by = "donor")
meta_df_C1$region = "epithelial cells"
rownames(meta_df_C1) <- meta_df_C1$cell_id
meta_df_C1$cell_id <- NULL
meta_df_C1<- meta_df_C1[colnames(expr_mat_C1), ]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/mammary_gland_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Nguyen_C1", as.matrix(expr_mat_C1), meta_df_C1, datasets_meta, cell_ontology)
message("Done!")
