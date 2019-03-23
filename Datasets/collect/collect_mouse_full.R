#! /usr/bin/env Rscript
# by caozj
# 12 Mar 2018
# 8:22:30 PM

# This script collects data from the mouse atlas


suppressPackageStartupMessages({
    library(Matrix)
    library(Matrix.utils)
    library(openxlsx)
    library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)


#===============================================================================
#
#  Read data
#
#===============================================================================

# Read cell labels
cat("Reading label file...\n")
label_df <- read.xlsx("../download/Mouse/MCA_cellassignments.xlsx",
                      "annotation")
rownames(label_df) <- label_df$Cell.name
label_df$Cell.name <- NULL


# Read DGE
cat("Reading DGE files...\n")
matrix_dir <- "../download/Mouse/MCA_rmbatch_dge_all"
combined_matrix <- NULL
matrix_files <- dir(matrix_dir)
idx <- 1
suppressWarnings({
    for (idx in 1:length(matrix_files)) {
        cat(sprintf(
            "[%2d/%2d] %s\r", idx, length(matrix_files), matrix_files[idx]
        ))
        matrix <- Matrix(as.matrix(read.table(
            file.path(matrix_dir, matrix_files[idx])
        )), sparse = TRUE)

        if (is.null(combined_matrix)) {
            combined_matrix <- matrix
        } else {
            expand_genes <- setdiff(rownames(matrix), rownames(combined_matrix))
            combined_matrix <- rbind(combined_matrix, Matrix(
                0, nrow = length(expand_genes), ncol = ncol(combined_matrix),
                sparse = TRUE, dimnames = list(expand_genes, NULL)
            ))  # Because merge.Matrix produces rows with name "fill.x"
                # for rows only observed in matrix y
            combined_matrix <- merge.Matrix(
                combined_matrix, matrix,
                by.x = rownames(combined_matrix), by.y = rownames(matrix)
            )
        }
    }
    cat("\n")
})

included_cells <- intersect(rownames(label_df), colnames(combined_matrix))
label_df <- label_df[included_cells, ]
combined_matrix <- combined_matrix[, included_cells]


#===============================================================================
#
#  Normalization and variable gene detection
#
#===============================================================================
so <- CreateSeuratObject(
    raw.data = combined_matrix, meta.data = label_df
)
so <- NormalizeData(
    object = so, normalization.method = "LogNormalize",
    scale.factor = 10000
)
pdf("./var_mouse_full.pdf")
so <- FindVariableGenes(
    object = so, mean.function = ExpMean, dispersion.function = LogVMR
)
dev.off()
cat(sprintf("Number of variable genes: %d\n", length(so@var.genes)))


#===============================================================================
#
#  Save result
#
#===============================================================================
write_dataset(so, "../data/Mouse_full/data.h5")
file.rename("./var_mouse_full.pdf",
            "../data/Mouse_full/var_genes.pdf")
cat("Done!\n")
