#! /usr/bin/env Rscript
# by caozj
# May 5, 2018
# 4:51:49 PM


suppressPackageStartupMessages({
    library(Matrix)
    library(Matrix.utils)
    library(SingleCellExperiment)
    library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)


#===============================================================================
#
#  Read data
#
#===============================================================================
message("Reading data...")
expr_mat <- readMM(
    "../download/PBMC/10xgenomics/jurkat_293t_50_50/filtered_matrices_mex/hg19/matrix.mtx"
)

gene_names <- read.table(
    "../download/PBMC/10xgenomics/jurkat_293t_50_50/filtered_matrices_mex/hg19/genes.tsv"
)$V2
expr_mat <- aggregate(expr_mat, groupings = gene_names, fun = "sum")
# This assigns row names as well

cell_names <- read.table(
    "../download/PBMC/10xgenomics/jurkat_293t_50_50/filtered_matrices_mex/hg19/barcodes.tsv"
)$V1
cdata <- read.csv(
    "../download/PBMC/10xgenomics/jurkat_293t_50_50/analysis_csv/kmeans/2_clusters/clusters.csv"
)
stopifnot(all(cdata$Barcode == cell_names))
rownames(cdata) <- cdata$Barcode
cdata$Barcode <- NULL
cdata$Cluster <- factor(cdata$Cluster, labels = c("293T", "Jurkat"))
colnames(expr_mat) <- rownames(cdata)


#===============================================================================
#
#  Normalization and variable gene detection
#
#===============================================================================
so <- CreateSeuratObject(
    raw.data = expr_mat, meta.data = cdata
)
so <- NormalizeData(
    object = so, normalization.method = "LogNormalize",
    scale.factor = 10000
)
pdf("./var_jurkat_293t.pdf")
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
write_dataset(so, "../data/Jurkat_293T/data.h5")
file.rename("./var_jurkat_293t.pdf",
            "../data/Jurkat_293T/var_genes.pdf")
message("Done!")
