#! /usr/bin/env Rscript
# by weil
# Aug 22, 2018
# 07:33 PM

suppressPackageStartupMessages({
    library(Matrix)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading 10x data...")
metadata <- read.csv("../download/Quake/annotations_droplet.csv",
                     header = TRUE, stringsAsFactors = FALSE)
cellid <- metadata[,1]
expr_mat <- NULL

samples <- dir("../download/Quake/droplet/")
for (sample in samples){
    mtx <- paste0("../download/Quake/droplet/", sample, "/matrix.mtx")
    expr <- readMM(mtx)

    barcode <- paste0("../download/Quake/droplet/", sample, "/barcodes.tsv")
    barcodes <- substring(read.csv(barcode, header =  FALSE)[,1], 1,16)
    channel <- strsplit(sample, "-")[[1]][2]
    barcodes <- paste(channel, barcodes, sep = "_")
    colnames(expr) <- barcodes

    gene <- paste0("../download/Quake/droplet/", sample, "/genes.tsv")
    genes <- read.delim(gene,header = FALSE)[,1]
    rownames(expr) <- genes

    cell.ann <- intersect(cellid, barcodes)
    expr.ann <- expr[,cell.ann]
    dense_expr <- as.matrix(expr.ann)
    expr_mat <- cbind(expr_mat, dense_expr)
}

meta_df <- metadata[, c(2:8,13:14)]
rownames(meta_df) <- cellid
colnames(meta_df) <- c("cell_type1", "cell_ontology_id", "plate", "cluster",
                       "free_annotation", "donor", "gender", "region", "organ")
meta_df$cell_ontology_class <- meta_df$cell_type1
meta_df$platform <- "10x"
meta_df$dataset_name <- "Quake_10x"
meta_df$organ <- gsub("Marrow", "Bone_Marrow", meta_df$organ)
meta_df <- meta_df[colnames(expr_mat), ]

#clean cell type
mask <- meta_df$cell_type1 != ""
meta_df <- meta_df[mask, ]
expr_mat <- expr_mat[, mask]

#clean ERCC
gene_mask_ERCC <- !grepl("ERCC-", rownames(expr_mat))
expr_mat <- expr_mat[gene_mask_ERCC, ]

construct_dataset(save_dir = "../data/Quake_10x", expr_mat, meta_df)

# take datasets apart by organ
# datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)

expr_10x <- read_dataset("../data/Quake_10x/data.h5")
cell_meta_10x <- expr_10x@obs
for (organ in unique(cell_meta_10x$organ)){
  mask <- cell_meta_10x$organ == organ
  expr_mat <- expr_10x@exprs[, mask]
  meta_df <- expr_10x@obs[mask, ]
  construct_dataset(save_dir = paste0("../data/Quake_10x_", organ), expr_mat, meta_df, datasets_meta, grouping = "donor")
}
message("Done!")
