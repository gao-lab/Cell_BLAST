#! /usr/bin/env Rscript
# by weil
# 7 Oct 2018
# 1:38 PM

# This script converts the RDS data downloaded from hemberg website
# into HDF5 format to be used by different methods
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sce <- readRDS("../download/Hemberg/Tasic/tasic-reads.rds")
#sce <- readRDS("../download/Hemberg/Tasic/tasic-rpkms.rds")
expr_mat <- as.matrix(counts(sce))
meta_df <- as.data.frame(colData(sce))
meta_df <- meta_df[, c("cluster_id", "major_class", "sub_class", "cell_type1", "cell_type2")]

# Clean up ERCC
mask <- grepl("ERCC-", rownames(expr_mat)) | rownames(expr_mat) == "tdTomato"
expr_mat <- expr_mat[!mask, ]

construct_dataset("../data/Tasic", expr_mat, meta_df)
message("Done!")

