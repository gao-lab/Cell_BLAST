#! /usr/bin/env Rscript
# by weil
# Dec 22, 2018
# 07:51 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_dropseq <- read.table("../download/Magella/GSM2796988_run1605_out_gene_exon_tagged.dge.txt.gz", 
                       sep = "\t", header = TRUE)
expr_dropseq <- expr_dropseq[!duplicated(expr_dropseq[,1]), ]
rownames(expr_dropseq) <- expr_dropseq[,1]
expr_dropseq[,1] <- NULL
expr_dropseq <- as.matrix(expr_dropseq)

expr_10x <- read.table("../download/Magella/GSM2796989_E14_5_WT_kidney_10X_matrix.txt.gz", 
                       sep = "\t", header = TRUE)
expr_10x <- expr_10x[!duplicated(expr_10x[,1]), ]
rownames(expr_10x) <- expr_10x[,1]
expr_10x[,1] <- NULL
expr_10x <- as.matrix(expr_10x)
colnames(expr_10x) <- gsub("\\.", "-", colnames(expr_10x))

expr_fluidigm <- read.table("../download/Magella/GSM2796990_E14_5_fluidigm_raw_894cells.txt.gz",
                    sep = "\t", header = TRUE)
expr_fluidigm <- expr_fluidigm[!duplicated(expr_fluidigm[,1]), ]
rownames(expr_fluidigm) <- expr_fluidigm[,1]
expr_fluidigm[,1] <- NULL
expr_fluidigm <- as.matrix(expr_fluidigm)

#metadata
meta_df <- read.csv("../download/Magella/cell_meta.csv", header = FALSE)
meta_df <- meta_df[, c(1,3)]
colnames(meta_df) <- c("cellID", "clusterID")
cluster_annotation <- read.csv("../download/Magella/Magella_cluster_annotation.csv")
meta_df <- merge(meta_df, cluster_annotation, by = "clusterID")
rownames(meta_df) <- meta_df$cellID
meta_df$cellID <- NULL
meta_df$organism = "Mus Musculus"
meta_df$organ = "Kidney"
meta_df$lifestage = "E14.5"

cell_dropseq <- intersect(colnames(expr_dropseq), rownames(meta_df))
meta_dropseq <- meta_df[cell_dropseq, ]
meta_dropseq$platform <- "Drop-seq"
meta_dropseq$dataset_name <- "Magella_dropseq"
expr_dropseq <- expr_dropseq[, cell_dropseq]

cell_10x <- intersect(colnames(expr_10x), rownames(meta_df))
meta_10x <- meta_df[cell_10x, ]
meta_10x$platform <- "10x"
meta_10x$dataset_name <- "Magella_10x"
expr_10x <- expr_10x[, cell_10x]

cell_fluidigm <- intersect(colnames(expr_fluidigm), rownames(meta_df))
meta_fluidigm <- meta_df[cell_fluidigm, ]
meta_fluidigm$platform <- "Fluidigm C1"
meta_fluidigm$dataset_name <- "Magella_fluidigm"
expr_fluidigm <- expr_fluidigm[, cell_fluidigm]

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/kidney_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Magella_dropseq", expr_dropseq, meta_dropseq, datasets_meta, cell_ontology)
construct_dataset("../data/Magella_10x", expr_10x, meta_10x, datasets_meta, cell_ontology)
construct_dataset("../data/Magella_fluidigm", expr_fluidigm, meta_fluidigm, datasets_meta, cell_ontology)
message("Done!")
