#! /usr/bin/env Rscript
# by weil
# 7 Oct 2018
# 1:51 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Usoskin/usoskin.rds")
expr_mat <- as.matrix(normcounts(sce))
meta_df <- as.data.frame(colData(sce))
colnames(meta_df)[4] <- "gender"

# Clean up ERCC
mask <- grepl("r_", rownames(expr_mat)) | rownames(expr_mat) == "tdTomato"
expr_mat <- expr_mat[!mask, ]

construct_dataset("../data/Usoskin", expr_mat, meta_df)
message("Done!")

