#! /usr/bin/env Rscript
# by weil
# Feb 6, 2020

suppressPackageStartupMessages({
  library(openxlsx)
  library(dplyr)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")

#meta_df
w9_meta <- read.csv("../download/Hochane/meta/w9_barcodes_celltypes.csv", header = TRUE)
w11_meta <- read.csv("../download/Hochane/meta/w11_barcodes_celltypes.csv", header = TRUE)
w13_meta <- read.csv("../download/Hochane/meta/w13_barcodes_celltypes.csv", header = TRUE)
w16_meta <- read.csv("../download/Hochane/meta/w16_barcodes_celltypes.csv", header = TRUE)
w18_meta <- read.csv("../download/Hochane/meta/w18_barcodes_celltypes.csv", header = TRUE)
colnames(w9_meta) <- c("barcode", "cell_type1")
colnames(w11_meta) <- c("barcode", "cell_type1")
colnames(w13_meta) <- c("barcode", "cell_type1")
colnames(w16_meta) <- c("barcode", "cell_type1")
colnames(w18_meta) <- c("barcode", "cell_type1")
w9_meta$donor <- "donor1"
w11_meta$donor <- "donor2"
w13_meta$donor <- "donor3"
w16_meta$donor <- "donor4"
w18_meta$donor <- "donor5"

w9_meta$age <- "9week"
w11_meta$age <- "11week"
w13_meta$age <- "13week"
w16_meta$age <- "16week"
w18_meta$age <- "18week"

#week 9 expr_mat
w9_mat <- readMM("../download/Hochane/GSM3509837_1_wk09_matrix.mtx.gz")
w9_col <- as.vector(read.table("../download/Hochane/GSM3509837_1_wk09_barcodes.tsv.gz", header = FALSE)[,1])
w9_row <- read.table("../download/Hochane/GSM3509837_1_wk09_genes.tsv.gz", header = FALSE)

#cell
w9_cell_mask <- match(w9_meta[,1], w9_col)
#gene
colnames(w9_row) <- c("gene_id", "symbol")
w9_gene_mask <- !duplicated(w9_row$symbol)
w9_row <- w9_row[w9_gene_mask, ]
rownames(w9_row) = w9_row$symbol
w9_row$symbol <- NULL

w9_mat <- w9_mat[w9_gene_mask, w9_cell_mask]
rownames(w9_mat) <- rownames(w9_row)
colnames(w9_mat) <- w9_meta[,1]

#week 11 expr_mat
w11_mat <- readMM("../download/Hochane/GSM3509838_2_wk11_matrix.mtx.gz")
w11_col <- as.vector(read.table("../download/Hochane/GSM3509838_2_wk11_barcodes.tsv.gz", header = FALSE)[,1])
w11_row <- read.table("../download/Hochane/GSM3509838_2_wk11_genes.tsv.gz", header = FALSE)

#cell
w11_cell_mask <- match(w11_meta[,1], w11_col)
#gene
colnames(w11_row) <- c("gene_id", "symbol")
w11_gene_mask <- !duplicated(w11_row$symbol)
w11_row <- w11_row[w11_gene_mask, ]

w11_mat <- w11_mat[w11_gene_mask, w11_cell_mask]
rownames(w11_mat) <- w11_row$symbol
colnames(w11_mat) <- w11_meta[,1]


#week 13 expr_mat
w13_mat <- readMM("../download/Hochane/GSM3509839_3_wk13_matrix.mtx.gz")
w13_col <- as.vector(read.table("../download/Hochane/GSM3509839_3_wk13_barcodes.tsv.gz", header = FALSE)[,1])
w13_row <- read.table("../download/Hochane/GSM3509839_3_wk13_genes.tsv.gz", header = FALSE)

#cell
w13_cell_mask <- match(w13_meta[,1], w13_col)
#gene
colnames(w13_row) <- c("gene_id", "symbol")
w13_gene_mask <- !duplicated(w13_row$symbol)
w13_row <- w13_row[w13_gene_mask, ]

w13_mat <- w13_mat[w13_gene_mask, w13_cell_mask]
rownames(w13_mat) <- w13_row$symbol
colnames(w13_mat) <- w13_meta[,1]

#week 16 expr_mat
w16_mat <- readMM("../download/Hochane/GSM3143601_4_w16_matrix.mtx.gz")
w16_col <- as.vector(read.table("../download/Hochane/GSM3143601_4_w16_barcodes.tsv.gz", header = FALSE)[,1])
w16_row <- read.table("../download/Hochane/GSM3143601_4_w16_genes.tsv.gz", header = FALSE)

#cell
w16_cell_mask <- match(w16_meta[,1], w16_col)
#gene
colnames(w16_row) <- c("gene_id", "symbol")
w16_gene_mask <- !duplicated(w16_row$symbol)
w16_row <- w16_row[w16_gene_mask, ]

w16_mat <- w16_mat[w16_gene_mask, w16_cell_mask]
rownames(w16_mat) <- w16_row$symbol
colnames(w16_mat) <- w16_meta[,1]

#week 18 expr_mat
w18_mat <- readMM("../download/Hochane/GSM3509840_5_wk18_matrix.mtx.gz")
w18_col <- as.vector(read.table("../download/Hochane/GSM3509840_5_wk18_barcodes.tsv.gz", header = FALSE)[,1])
w18_row <- read.table("../download/Hochane/GSM3509840_5_wk18_genes.tsv.gz", header = FALSE)

#cell
w18_cell_mask <- match(w18_meta[,1], w18_col)
#gene
colnames(w18_row) <- c("gene_id", "symbol")
w18_gene_mask <- !duplicated(w18_row$symbol)
w18_row <- w18_row[w18_gene_mask, ]

w18_mat <- w18_mat[w18_gene_mask, w18_cell_mask]
rownames(w18_mat) <- w18_row$symbol
colnames(w18_mat) <- w18_meta[,1]

#integrate
colnames(w9_mat) <- paste0(colnames(w9_mat), "_w9")
colnames(w11_mat) <- paste0(colnames(w11_mat), "_w11")
colnames(w13_mat) <- paste0(colnames(w13_mat), "_w13")
colnames(w16_mat) <- paste0(colnames(w16_mat), "_w16")
colnames(w18_mat) <- paste0(colnames(w18_mat), "_w18")

expr_mat <- cbind(w9_mat, w11_mat, w13_mat, w16_mat, w18_mat)
meta_df <- rbind(w9_meta, w11_meta, w13_meta, w16_meta, w18_meta)
rownames(meta_df) <- colnames(expr_mat)
meta_df$barcode <- NULL
meta_df$cell_id <- rownames(meta_df)

# filter cell
meta_df$cell_type1 <- as.character(meta_df$cell_type1)
meta_df_filtered <- meta_df %>%
  filter(
    cell_type1 != "DTLH",
    cell_type1 != "RVCSBa",
    cell_type1 != "RVCSBb",
    cell_type1 != "SSBm/d",
    cell_type1 != "UBCD",
  )
rownames(meta_df_filtered) <- meta_df_filtered$cell_id
meta_df_filtered$cell_id <- NULL
cell_mask <- match(rownames(meta_df_filtered), meta_df$cell_id)
expr_mat_filtered <- expr_mat[, cell_mask]

# Assign cell ontology
cell_ontology <- read.csv("../cell_ontology/kidney_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets_2.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Hochane", expr_mat_filtered, meta_df_filtered, datasets_meta=datasets_meta, 
                  cell_ontology=cell_ontology, gene_meta = w9_row, grouping = "donor")
#, min_group_frac = 0.25)
message("Done!")
