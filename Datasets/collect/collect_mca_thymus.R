#! usr/bin/env Rscript
# by weil

source("../../Utilities/data.R", chdir = TRUE)

expr_mat1 <- read.table("../download/MCA/500more_dge/Thymus1_dge.txt.gz", sep = " ",
                        row.names = 1, header = TRUE)
#no overlap with meta_df
#expr_mat2 <- read.table("../download/MCA/500more_dge/Thymus2_dge.txt.gz", sep = " ",
#                        row.names = 1, header = TRUE)
meta_df <- read.csv("../download/MCA/MCA_CellAssignments_new.csv", stringsAsFactors = FALSE)
rownames(meta_df) <- meta_df$Cell.name
meta_df <- meta_df[, c("ClusterID", "Annotation")]
colnames(meta_df) <- c("cluster", "cell_type1")
meta_df$dataset_name <- "MCA_Thymus"
meta_df$platform <- "Microwell-Seq"
meta_df$organism <- "Mus musculus"
meta_df$organ <- "Thymus"

thymus_cell <- intersect(colnames(expr_mat1), rownames(meta_df))
expr_mat <- expr_mat1[, thymus_cell]
meta_df <- meta_df[thymus_cell, ]

construct_dataset("../data/MCA_thymus", as.matrix(expr_mat), meta_df)
message("Done!")
