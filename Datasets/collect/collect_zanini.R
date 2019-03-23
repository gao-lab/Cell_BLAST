#! /usr/bin/env Rscript
# by caozj

suppressPackageStartupMessages({
    library(dplyr)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- read.table("../download/Zanini/counts_10_10_unique_L1.tsv.gz",
                       sep = "\t", header = TRUE, row.names = 1, check.names = FALSE)
cell_meta <- read.table("../download/Zanini/samplesheet_10_10_unique_L1.tsv",
                        sep = "\t", header = TRUE, comment.char = "", quote = "")
gene_meta <- read.table("../download/Zanini/featuresheet_10_10_unique_L1.tsv",
                        sep = "\t", header = TRUE, comment.char = "", quote = "")
expr_mat <- as.matrix(expr_mat)

cell_mask <- cell_meta$cellType != "unknown"
cell_meta <- cell_meta[cell_mask, ]
expr_mat <- expr_mat[, cell_mask]
stopifnot(all(rownames(expr_mat) == gene_meta$EnsemblGeneID))
rownames(expr_mat) <- gene_meta$GeneName
rownames(gene_meta) <- gene_meta$GeneName
gene_meta$GeneName <- NULL

cell_meta <- cell_meta %>% dplyr::select(
    cell_id = name, cell_type1 = cellType, donor = patient,
    seq_run = sequencingRun, seq_instrument = sequencingInstrument
)
infection <- read.csv("../download/Zanini/infection.csv")
cell_meta <- merge(cell_meta, infection, by="donor")
cell_meta$organism <- "Homo sapiens"
cell_meta$organ <- "PBMC"
cell_meta$platform <- "viscRNA-Seq"
cell_meta$dataset_name <- "Zanini"
rownames(cell_meta) <- cell_meta$cell_id
cell_meta$cell_id <- NULL

construct_dataset("../data/Zanini", expr_mat, cell_meta, gene_meta = gene_meta, grouping = "donor")
message("Done!")
