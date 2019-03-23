#! /usr/bin/env Rscript
# by caozj
# 27 Feb 2018
# 7:54:40 PM


suppressPackageStartupMessages({
    library(Seurat)
    library(rhdf5)
})
source("../../Utilities/data.R", chdir = TRUE)


cat("Reading data...\n")
pure <- readRDS("../download/PBMC/rds/all_pure_pbmc_data.rds")
pure_idx <- readRDS("../download/PBMC/rds/pure_idx.rds")

combined_matrix <- list()
combined_cell_meta <- list()
for (cell_type in names(pure_idx)) {
    sample_idx <- unique(pure_idx[[cell_type]]$sample)
    stopifnot(length(sample_idx) == 1)
    this_mat <- pure$all_data[[sample_idx]]$hg19$mat
    rownames(this_mat) <- paste(
        sample_idx, pure$all_data[[sample_idx]]$hg19$barcode, sep = "-"
    )
    colnames(this_mat) <- pure$all_data[[sample_idx]]$hg19$gene_symbols
    combined_matrix[[cell_type]] <- t(this_mat[
        pure_idx[[cell_type]]$use,
        !duplicated(colnames(this_mat))
    ])  # TODO: now keeps only the first one among duplicated genes
    combined_cell_meta[[cell_type]] <- data.frame(
        cell_type1 = rep(cell_type, ncol(combined_matrix[[cell_type]])),
        row.names = colnames(combined_matrix[[cell_type]])
    )
}
combined_matrix <- Reduce(cbind, combined_matrix)
cell_meta <- Reduce(rbind, combined_cell_meta)

known_markers <- c(
    "CD4",
    "PTPRC",  # CD45
    "IL2RA",  # CD25
    "NCAM1",  # CD56
    "CD8A",
    "CD14",
    "CD8B",
    "CD34",
    "CD19",
    "CLEC9A"
)

cell_ontology <- read.csv("../cell_ontology/blood_cell_ontology.csv")

datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset(
    "../data/Zheng", combined_matrix, cell_meta, datasets_meta, cell_ontology,
    gene_list = list(
        expressed_genes = known_markers,
        seurat_genes = known_markers,
        scmap_genes = known_markers,
        known_markers = known_markers
    ), y_low = 0
)
cat("Done!\n")
