#! /usr/bin/env Rscript
# by caozj

suppressPackageStartupMessages({
    library(dplyr)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
data <- read.csv("../download/Tusi/GSM2388072_basal_bone_marrow.raw_umifm_counts.csv.gz", check.names = FALSE)
expr_mat <- as.matrix(data[, 6:ncol(data)])
rownames(expr_mat) <- data$cell_id
meta_df <- data[, 1:6] %>% filter(pass_filter == 1) %>% select(
    cell_id, batch = library_id
)
fate <- read.csv(
    "../download/Tusi/Supplementary Data/PBA_inputs/bBM/text_file_output/B.csv",
    header = FALSE
)
fate_labels <- read.csv("../download/Tusi/Supplementary Data/PBA_inputs/bBM/fate_labels.csv")
colnames(fate) <- colnames(fate_labels)
potential <- read.csv(
    "../download/Tusi/Supplementary Data/PBA_inputs/bBM/text_file_output/V.txt",
    header = FALSE
)
colnames(potential) <- "potential"
meta_df <- Reduce(cbind, list(meta_df, fate, potential))
rownames(meta_df) <- meta_df$cell_id
meta_df$cell_id <- NULL
meta_df$cell_type1 = "HSPC"

expr_mat <- expr_mat[rownames(meta_df), ]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/bone_marrow_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Tusi", t(expr_mat), meta_df, datasets_meta, cell_ontology, grouping = "batch")
message("Done!")
