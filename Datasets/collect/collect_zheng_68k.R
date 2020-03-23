#! /usr/bin/env Rscript
# by caozj
# 27 Feb 2018
# 7:54:40 PM


suppressPackageStartupMessages({
    library(Seurat)
    library(rhdf5)
    library(Matrix)
    library(ggplot2)
    library(Rtsne)
    library(svd)
    library(dplyr)
    library(plyr)
    library(data.table)
    library(pheatmap)
})
source("../../Utilities/data.R")


cat("Reading data...\n")
pbmc_68k <- readRDS("../download/PBMC/rds/pbmc68k_data.rds")
author_genes <- read.table(
    "../download/PBMC/single-cell-3prime-paper/pbmc68k_analysis/author_genes.txt",
    stringsAsFactors = FALSE
)[, 1]
author_annotation <- read.table(
    "../download/PBMC/single-cell-3prime-paper/pbmc68k_analysis/author_annotation.txt",
    stringsAsFactors = FALSE, header = TRUE
)

exprs <- t(pbmc_68k$all_data[[1]]$hg19$mat)
rownames(exprs) <- pbmc_68k$all_data[[1]]$hg19$genes
colnames(exprs) <- 1:ncol(exprs)

cell_df <- data.frame(cell_type1 = author_annotation$cls_id,
                      row.names = 1:nrow(author_annotation),
                      stringsAsFactors = FALSE)
gene_df <- data.frame(gene_symbol = pbmc_68k$all_data[[1]]$hg19$gene_symbols,
                      row.names = pbmc_68k$all_data[[1]]$hg19$genes)

known_markers <- c(
    "ENSG00000010610",  # CD4
    "ENSG00000081237",  # CD45
    "ENSG00000134460",  # CD25
    "ENSG00000149294",  # CD56
    "ENSG00000153563",  # CD8A
    "ENSG00000170458",  # CD14
    "ENSG00000172116",  # CD8B
    "ENSG00000174059",  # CD34
    "ENSG00000177455",  # CD19
    "ENSG00000197992"   # CLEC9A
)
construct_dataset(
    "../data/PBMC_68k", exprs, cell_df, gene_meta = gene_df,
    gene_list = list(
        expressed_genes = known_markers,
        seurat_genes = known_markers,
        scmap_genes = known_markers,
        known_markers = known_markers,
        author_genes = author_genes
    ), y_low = 0
)
cat("Done!\n")

# Above is previous version
# Here add cell ontology
data=read_dataset("../data/PBMC_68k/data.h5")

expr_mat=data@exprs
meta_df=data.frame(data@obs)
gene_meta = data@var
gene_meta$gene_id=rownames(gene_meta)
# duplicated gene
mask=!duplicated(gene_meta$gene_symbol)
expr_mat = expr_mat[mask,]
gene_meta=gene_meta[mask,]
rownames(gene_meta)=gene_meta$gene_symbol
rownames(expr_mat)=gene_meta$gene_symbol
#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/blood_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
#datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Zheng_68k", expr_mat, cell_meta = meta_df, gene_meta = gene_meta,
                  cell_ontology = cell_ontology)
message("Done!")
