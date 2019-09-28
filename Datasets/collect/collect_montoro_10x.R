#! /usr/bin/env Rscript
# by caozj
# Dec 6, 2018
# 4:08:48 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.table(
    "../download/Montoro/GSE103354_Trachea_droplet_UMIcounts.txt.gz",
    header = TRUE
)
cell_name_split <- strsplit(colnames(expr_mat), "_")
meta_df <- data.frame(
    donor = sapply(cell_name_split, function(x) x[1]),  # 4 WT, 2 Foxj1-EGFP
    cell_type1 = sapply(cell_name_split, function(x) x[3]),  # 7 cell types
    row.names = colnames(expr_mat)
)

cell_ontology <- read.csv("../cell_ontology/trachea_cell_ontology.csv")

# datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Montoro_10x", as.matrix(expr_mat), meta_df,
                  datasets_meta = datasets_meta, cell_ontology = cell_ontology, grouping = "donor", y_low = 0.8)

mask <- meta_df$cell_type1 != "Ionocyte"

construct_dataset("../data/Montoro_10x_noi", as.matrix(expr_mat[, mask]), meta_df[mask, ],
                  datasets_meta = datasets_meta, cell_ontology = cell_ontology, grouping = "donor", y_low = 0.8)
message("Done!")
