#! /usr/bin/env Rscript
# by weil
# 7 Oct 2018
# 9:20 AM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Kolodziejczyk/kolodziejczyk.rds")
expr_mat <- as.matrix(counts(sce))
meta_df <- as.data.frame(colData(sce))
meta_df <- meta_df[, c("cell_type1", "batch")]

# Clean up ERCC
mask <- grepl("ERCC-", rownames(expr_mat))
expr_mat <- expr_mat[!mask, ]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Kolodziejczyk", expr_mat, meta_df, datasets_meta)
message("Done!")

