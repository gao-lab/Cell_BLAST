#! /usr/bin/env Rscript
# by weil
# Aug 26, 2018
# 03:22 PM

#suppressPackageStartupMessages({
#    library(Matrix)
#})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading FACS data...")
metadata_facs <- read.csv("../download/Quake/annotations_FACS.csv", stringsAsFactors = FALSE)
cell_facs <- metadata_facs[, "cell"]
expr_mat  <- NULL

samples_facs <- dir("../download/Quake/FACS/")
for (sample in samples_facs){
    counts <- paste0("../download/Quake/FACS/", sample)
    expr_facs <- read.csv(counts)
    rownames(expr_facs) <- expr_facs[,1]
    expr_facs[,1] <- NULL
#    expr_facs <- expr_facs[rownames(expr_mat_10x),]
    expr_facs <- as.matrix(expr_facs[, intersect(colnames(expr_facs), cell_facs)])
    expr_mat <- cbind(expr_mat, expr_facs)
}

meta_df <- metadata_facs[, c(4:10, 21:22)]
rownames(meta_df) <- cell_facs
colnames(meta_df) <- c("cell_type1", "cell_ontology_id", "cluster", "free_annotation",
                       "donor", "gender", "channel", "region", "organ")
meta_df$cell_ontology_class <- meta_df$cell_type1
meta_df$platform <- "Smart-seq2"
meta_df <- meta_df[colnames(expr_mat),]
meta_df <- meta_df[,c(1:6, 8:11,7)]
meta_df$dataset_name <- "Quake_Smart-seq2"
meta_df$organ <- gsub("Marrow", "Bone_Marrow", meta_df$organ)
meta_df <- meta_df[colnames(expr_mat), ]

#clean cell type
mask <- meta_df$cell_type1 != ""
meta_df <- meta_df[mask, ]
expr_mat <- expr_mat[, mask]

#clean ERCC
gene_mask <- !grepl("ERCC-", rownames(expr_mat))
expr_mat <- expr_mat[gene_mask, ]

construct_dataset(save_dir = "../data/Quake_Smart-seq2", expr_mat, meta_df)

# take datasets apart by organ
# datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)

expr_ss2 <- read_dataset("../data/Quake_Smart-seq2/data.h5")
cell_meta_ss2 <- expr_ss2@obs
for (organ in unique(cell_meta_ss2$organ)){
  mask <- cell_meta_ss2$organ == organ
  expr_mat <- expr_ss2@exprs[, mask]
  meta_df <- expr_ss2@obs[mask, ]
  construct_dataset(save_dir = paste0("../data/Quake_Smart-seq2_", organ), expr_mat, meta_df, datasets_meta, grouping = "donor")
}
message("Done!")
