#! /usr/bin/env Rscript
# by weil
# Mar 06, 2019
# 02:44 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.table("../download/Farrell/GSE106474_UMICounts.txt.gz")
expr_mat <- as.matrix(expr_mat)

pseudotime <- read.table("../download/Farrell/dropseq_cluster_URDDevelopmentaltree.txt", row.names = 1)
pseudotime <- pseudotime[3:dim(pseudotime)[1], 3, drop = FALSE]
colnames(pseudotime) <- "probabilistic_breadth"
expr_atlas <- expr_mat[, rownames(pseudotime)]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Farrell", expr_atlas, pseudotime, datasets_meta)
message("Done!")
