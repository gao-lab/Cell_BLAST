#! /usr/bin/env Rscript
# by caozj
# 12 Mar 2018
# 8:22:30 PM

# This script collects data from the mouse atlas


suppressPackageStartupMessages({
    library(Matrix)
    library(Matrix.utils)
    library(openxlsx)
    library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)


# Read cell labels
cat("Reading label file...\n")
cell_df <- read.xlsx("../download/Mouse/MCA_Figure2_Cell.Info.xlsx",
                     "MCA_Fig2_Cell")
colnames(cell_df)[1] <- "CellID"
cell_type_df <- read.xlsx("../download/Mouse/MCA_Figure2_Cell.Info.xlsx",
                          "MCA_Fig2_Celltype")
cell_type_df$Cluster.Name <- ave(
    cell_type_df$Cell.Type, cell_type_df$Cell.Type, FUN = function(x) {
        if (length(x) == 1) {
            return(x)
        } else {
            return(sprintf("%s (%d)", x, seq_along(x)))
        }
    }
)
merged_cell_df <- merge(
    cell_df, cell_type_df, by.x = "ClusterID", by.y = "Cluster",
    all.x = TRUE, sort = FALSE
)
rownames(merged_cell_df) <- merged_cell_df$CellID
merged_cell_df$CellID <- NULL

# Read DGE
cat("Reading DGE...\n")
exprs <- Matrix(as.matrix(read.table(
    "../download/Mouse/MCA_Figure2_batchremoved.txt.gz"
)), sparse = TRUE)

included_cells <- intersect(rownames(merged_cell_df), colnames(exprs))
merged_cell_df <- merged_cell_df[included_cells, ]
exprs <- exprs[, included_cells]
expressed_genes <- rownames(exprs)[rowSums(exprs > 1) > 5]

merged_cell_df <- merged_cell_df[, c(
    "Cell.Type", "Cluster.Name", "Tissue"
)]
colnames(merged_cell_df) <- c(
    "cell_type1", "cell_type2", "tissue"
)

message("Constructing dataset...")
dataset <- new("ExprDataSet",
    exprs = exprs, obs = merged_cell_df,
    var = data.frame(row.names = rownames(exprs)),
    uns = list(expressed_genes = expressed_genes)
)

message("Saving data...")
write_dataset(dataset, "../data/Mouse_60k/data.h5")
cat("Done!\n")
