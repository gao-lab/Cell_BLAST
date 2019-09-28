#! /usr/bin/env Rscript
# by caozj
# Dec 6, 2018
# 6:54:11 PM

source("../../Utilities/data.R", chdir = TRUE)
library(dplyr)

message("Reading data...")
meta_df <- read.table(
    "../download/Plasschaert/GSE102580_meta_filtered_counts_mouse.tsv.gz",
    header = TRUE, sep = "\t", row.names = 1
)
meta_df <- meta_df[!is.na(meta_df$x_Fig1c), ]

files <- list.files("../download/Plasschaert")
files <- files[grepl("[uU]ninjured_mouse_id", files)]
raw_data_files <- list()
colnames_sanity <- NULL
for (file in files) {
    expr_mat <- read.table(
        file.path("../download/Plasschaert", file), header = TRUE, check.names = FALSE)
    expr_mat$raw_data_file <- rep(paste(
        strsplit(file, "_")[[1]][-1], collapse = "_"
    ), nrow(expr_mat))
    if (length(raw_data_files) == 0) {
        colnames_sanity <- colnames(expr_mat)
    } else {
        stopifnot(all(colnames(expr_mat) == colnames_sanity))
    }
    raw_data_files[[length(raw_data_files) + 1]] <- expr_mat
    message(file)
}
expr_mat <- do.call(rbind, raw_data_files)

expr_mat_with_meta <- merge(
    expr_mat, meta_df, by = c("barcode", "raw_data_file"))
meta_mask <- colnames(expr_mat_with_meta) %in% colnames(meta_df)
new_meta_df <- expr_mat_with_meta[, meta_mask]
new_expr_mat <- expr_mat_with_meta[, !meta_mask]

new_meta_df <- new_meta_df[, c(
    "unique_library_id", "mouse_id", "timepoint", "UMIFM", "fraction_mito",
    "clusters_Fig1", "clusters_Fig2"
)]
colnames(new_meta_df) <- c(
    "library", "donor", "disease", "umi_count", "mito",
    "cell_type1", "cell_type2"
)

rownames(new_meta_df) <- as.character(1:nrow(new_meta_df))
rownames(new_expr_mat) <- as.character(1:nrow(new_expr_mat))

cell_ontology <- read.csv("../cell_ontology/trachea_cell_ontology.csv")

# datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)

construct_dataset("../data/Plasschaert",
    t(as.matrix(new_expr_mat)), new_meta_df,
    datasets_meta = datasets_meta, cell_ontology = cell_ontology
)

mask <- new_meta_df$cell_type1 != "Ionocytes"
construct_dataset("../data/Plasschaert_noi",
    t(as.matrix(new_expr_mat[mask, ])), new_meta_df[mask, ],
    datasets_meta = datasets_meta, cell_ontology = cell_ontology
)
message("Done!")
