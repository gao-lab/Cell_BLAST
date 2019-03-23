#! /usr/bin/env Rscript
# by caozj

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- as.matrix(read.table(
    "../download/Park/GSE107585_Mouse_kidney_single_cell_datamatrix.txt.gz"))
meta_df <- as.data.frame(expr_mat[1, ])
colnames(meta_df) <- "cluster"
meta_df$barcode <- rownames(meta_df)
meta_df$donor <- paste("donor", sapply(
    strsplit(meta_df$barcode, "\\."),
    function(x) x[2]
), sep = "_")
expr_mat <- expr_mat[2:nrow(expr_mat), ]

cluster_map <- read.csv("../download/Park/cluster_map.csv")
meta_df <- merge(meta_df, cluster_map)
meta_df$cluster <- NULL
rownames(meta_df) <- meta_df$barcode
meta_df$barcode <- NULL
meta_df <- meta_df[colnames(expr_mat), , drop = FALSE]

meta_df$organism <- "Mus musculus"
meta_df$organ <- "kidney"
meta_df$platform <- "10x"
meta_df$dataset_name <- "Park"

cell_ontology <- read.csv("../cell_ontology/kidney_cell_ontology.csv")

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Park", expr_mat, meta_df, datasets_meta, cell_ontology,
                  binning = "equal_width", x_low = 0.0125, x_high = 3.0, y_low = 0.5,
                  grouping = "donor")
message("Done!")
