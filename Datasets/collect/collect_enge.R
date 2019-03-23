#! /usr/bin/env Rscript
# by caozj
# May 23, 2018
# 3:52:59 PM

suppressPackageStartupMessages({
    library(GEOquery)
    library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

message("Reading data...")
data_path <- "../download/Enge"
es <- getGEO(filename = file.path(data_path, "GSE81547_series_matrix.txt.gz"))
meta_df <- pData(es) %>% dplyr::select(
    lifestage = `donor_age:ch1`,
    gender = `gender:ch1`,
    cell_type1 = `inferred_cell_type:ch1`
)
# colnames(meta_df) <- c(
#     "age", "gender", "cell_type1"
# )
meta_df$lifestage <- as.integer(meta_df$lifestage)
meta_df$donor <- paste("donor", as.character(as.integer(factor(meta_df$lifestage))), sep = "_")

data_list <- list()
gene_order <- NULL
for (data_file in dir(data_path, pattern = ".*\\.csv\\.gz")) {
    cat(sprintf("Reading %s...\r", data_file))
    gsm <- strsplit(data_file, "_")[[1]][1]
    this_cell <- read.table(file.path(data_path, data_file),
                            row.names = 1)
    if (is.null(gene_order))
        gene_order <- rownames(this_cell)
    data_list[[gsm]] <- this_cell[gene_order, 1]
}
cat("\n")
count_mat <- as.matrix(as.data.frame(data_list))
rownames(count_mat) <- gene_order

message("Filtering...")

# Clean non-genes
gene_mask <- gene_order %in% c(
    "no_feature", "ambiguous",
    "too_low_aQual", "not_aligned",
    "alignment_not_unique"
) | grepl("^ERCC-", gene_order)
count_mat <- count_mat[!gene_mask, ]

# Clean cell types
mask <- meta_df$cell_type1 != "unsure"
expr_mat <- count_mat[, mask]
meta_df <- meta_df[mask, ]

cell_ontology <- read.csv("../cell_ontology/pancreas_cell_ontology.csv")
#datasets_meta
datasets_meta <- read.csv("../ACA_datasets.csv", header = TRUE, row.names = 1)
construct_dataset("../data/Enge", expr_mat, meta_df, datasets_meta, cell_ontology, grouping = "donor", min_group_frac = 0.4)

message("Done!")
