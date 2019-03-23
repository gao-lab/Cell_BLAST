#! /usr/bin/env Rscript
# by caozj
# Jul 31, 2018
# 5:18:50 PM

suppressPackageStartupMessages({
    library(openxlsx)
    library(rhdf5)
    library(dplyr)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading matrix...")
root <- "../download/Planarian"
principal <- as.matrix(read.table(file.path(root,
    "GSE111764_PrincipalClusteringDigitalExpressionMatrix.dge.txt.gz"
)))
# brain <- as.matrix(read.table(file.path(root,
#     "GSE111764_BrainClusteringDigitalExpressionMatrix.dge.txt.gz"
# )))  # No major cluster assignment
# sexual <- read.table(file.path(root,
#     "GSE111764_SexualClusteringDigitalExpressionMatrix.dge.txt.gz"
# ))  # Sexual strain, gene annotations are different

message("Reading meta table...")
meta_df <- read.xlsx(file.path(root,
    "aaq1736_Table-S1.xlsx"
), sheet = 1, startRow = 4, rowNames = TRUE)
meta_df <- meta_df[, c(
    "Section", "FACS_Gate", "Cluster.ID", "Major.cluster.description", "Subcluster.ID"
)]
colnames(meta_df) <- c("organ", "FACS_gate", "cluster", "cell_type1", "free_annotation")

message("Cleaning and merging...")
# common_genes <- intersect(rownames(principal), rownames(brain))
# principal <- principal[common_genes, ]
# brain <- brain[common_genes, ]
# mat <- cbind(principal, brain)
mat <- principal  # Compatability
common_cells <- intersect(colnames(mat), rownames(meta_df))
mat <- mat[, common_cells]
meta_df <- meta_df[common_cells, ]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/planaria_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Fincher", mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
