#! /usr/bin/env Rscript
# by caozj
# Jul 18, 2018
# 9:56:30 AM

suppressPackageStartupMessages({
    library(argparse)
    library(rhdf5)
    library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

parser <- ArgumentParser()
parser$add_argument("-d", "--data", dest = "data", type = "character", required = TRUE)
parser$add_argument("-u", "--uns-slot", dest = "uns_slot", type = "character", default = "seurat_genes")
parser$add_argument("--x-low", dest = "x_low", type = "double", default = 0.1)
parser$add_argument("--x-high", dest = "x_high", type = "double", default = 8)
parser$add_argument("--y-low", dest = "y_low", type = "double", default = 1)
parser$add_argument("--y-high", dest = "y_high", type = "double", default = NULL)
args <- parser$parse_args()

if (is.null(args$x_high))
    args$x_high <- Inf
if (is.null(args$x_low))
    args$x_low <- -Inf
if (is.null(args$y_high))
    args$y_high <- Inf
if (is.null(args$y_low))
    args$y_low <- -Inf

dataset <- read_dataset(args$data)
so <- to_seurat(dataset)

so <- NormalizeData(
    object = so, normalization.method = "LogNormalize",
    scale.factor = 10000
)

pdf(sprintf("%s/%s.pdf", dirname(args$data), args$uns_slot))
so <- FindVariableGenes(
    object = so, mean.function = ExpMean, dispersion.function = LogVMR,
    x.low.cutoff = args$x_low, x.high.cutoff = args$x_high,
    y.cutoff = args$y_low, y.high.cutoff = args$y_high,
    binning.method = "equal_frequency"
)
dev.off()
cat(sprintf("Number of variable genes: %d\n", length(so@var.genes)))

write_hybrid_path(so@var.genes, sprintf(
    "%s//uns/%s", args$data, args$uns_slot))

message("Done!")
