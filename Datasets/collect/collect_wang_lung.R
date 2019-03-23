#! /usr/bin/env Rscript
# by weil
# Dec 30, 2018
# 07:46 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat1 <- read.csv("../download/Wang_lung/GSM2858339_AT1_P3.exprs.csv.gz", row.names = 1)
colnames(expr_mat1) <- paste0(colnames(expr_mat1), "_1")
expr_mat2 <- read.csv("../download/Wang_lung/GSM2858340_AT1_P15.exprs.csv.gz", row.names = 1)
colnames(expr_mat2) <- paste0(colnames(expr_mat2), "_2")
expr_mat3 <- read.csv("../download/Wang_lung/GSM2858341_AT1_P60.exprs.csv.gz", row.names = 1)
colnames(expr_mat3) <- paste0(colnames(expr_mat3), "_3")
expr_mat4 <- read.csv("../download/Wang_lung/GSM2858342_AT2_P60.exprs.csv.gz", row.names = 1)
colnames(expr_mat4) <- paste0(colnames(expr_mat4), "_4")

expr_mat12 <- merge(expr_mat1, expr_mat2, by = 0, all = TRUE)
expr_mat34 <- merge(expr_mat3, expr_mat4, by = 0, all = TRUE)
expr_mat <- merge(expr_mat12, expr_mat34, by = "Row.names", all = TRUE)
rownames(expr_mat) <- expr_mat$Row.names
expr_mat$Row.names <- NULL
expr_mat <- as.matrix(expr_mat)
expr_mat[is.na(expr_mat)] <- 0


meta_df1 <- data.frame(row.names = colnames(expr_mat1))
meta_df1$cell_type1 = "alveolar type 1"
meta_df1$lifestage = "postnatal day 3"

meta_df2 <- data.frame(row.names = colnames(expr_mat2))
meta_df2$cell_type1 = "alveolar type 1"
meta_df2$lifestage = "postnatal day 15"

meta_df3 <- data.frame(row.names = colnames(expr_mat3))
meta_df3$cell_type1 = "alveolar type 1"
meta_df3$lifestage = "postnatal day 60"

meta_df4 <- data.frame(row.names = colnames(expr_mat4))
meta_df4$cell_type1 = "alveolar type 2"
meta_df4$lifestage = "postnatal day 60"

meta_df <- rbind(meta_df1, meta_df2, meta_df3, meta_df4)
meta_df$region <- "pulmonary alveolar epithelium"
meta_df <- meta_df[colnames(expr_mat), ]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/lung_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Wang_Lung", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
