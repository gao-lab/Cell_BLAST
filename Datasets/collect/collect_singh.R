#! /usr/bin/env Rscript
# by weil
# Dec 26, 2018
# 08:25 PM

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
expr_mat <- as.matrix(read.csv("../download/Singh/GSE109881_Zf_GERASStages_Counts.csv.gz", header = TRUE, row.names = 1))
gene_meta <- data.frame(gene_id = rownames(expr_mat), stringsAsFactors = FALSE)
ensembl_89 <- read.csv("../../../GENOME/mapping/zebrafish_89.txt", stringsAsFactors = FALSE)
gene_meta <- merge(gene_meta, ensembl_89, by.x = "gene_id", by.y = "Gene.stable.ID", all = FALSE)
gene_meta <- gene_meta[!duplicated(gene_meta$Gene.name), ]
expr_mat <- expr_mat[gene_meta$gene_id, ]
rownames(expr_mat) <- gene_meta$Gene.name
rownames(gene_meta) <- gene_meta$Gene.name
gene_meta$Gene.name <- NULL

meta_df <- data.frame(row.names = colnames(expr_mat))
meta_df$lifestage <- unlist(lapply(strsplit(colnames(expr_mat), "_"), function(x) x[1])) 
meta_df$cell_type1 = "beta"
meta_df$organism = "Danio rerio"
meta_df$organ = "Pancreas"
meta_df$platform = "Smart-seq2"
meta_df$dataset_name = "Singh"

#assign cell ontology
cell_ontology <- read.csv("../cell_ontology/pancreas_cell_ontology.csv")
cell_ontology <- cell_ontology[, c("cell_type1", "cell_ontology_class", "cell_ontology_id")]

#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Singh", expr_mat, meta_df, datasets_meta, cell_ontology)
message("Done!")
