#! /usr/bin/env Rscript
# by weil
# Dec 26, 2018
# 03:41 PM

suppressPackageStartupMessages({
  library(openxlsx)
})

source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
sample <- dir("../download/Xin_2018/", pattern = "mtx")
file_path <- "../download/Xin_2018"
expr_mat <- NULL
for (i in 1:12){
  expr_mat <- cbind(expr_mat, readMM(file.path(file_path, sample[i])))
}

gene <- read.table("../download/Xin_2018/GSM3138939_Donor_1_genes.tsv.gz", sep = "\t", stringsAsFactors = FALSE)
dup <- duplicated(gene[,2])
expr_mat <- expr_mat[!dup,]
gene <- gene[!dup, 2]
rownames(expr_mat) <- gene

barcodes <- dir("../download/Xin_2018/", pattern = "barcodes")
cell_id <- NULL
for (i in 1:12){
  donor <- unlist(strsplit(barcodes[i], "_"))[3]
  paste0(donor, "_", read.table(file.path(file_path, barcodes[i]))[,1])
  cell_id <- c(cell_id, paste0(donor, "_", read.table(file.path(file_path, barcodes[i]))[,1]))
}
colnames(expr_mat) <- cell_id

#meta_df without cell_type1
meta_df <- data.frame(cell_id = cell_id)
meta_df$donor <- unlist(lapply(strsplit(colnames(expr_mat), "_"), function(x) x[1]))
donor_annotation <- read.xlsx("../download/Xin_2018/DB180365SupplementaryTableS1.xlsx")[, 1:4]
colnames(donor_annotation) <- c("donor", "age", "country", "gender")
meta_df <- merge(meta_df, donor_annotation, by = "donor")
rownames(meta_df) <- meta_df$cell_id
meta_df$cell_id <- NULL
meta_df <- meta_df[colnames(expr_mat), ]

construct_dataset("../data/Xin_2018", expr_mat, meta_df)
message("Done!")
