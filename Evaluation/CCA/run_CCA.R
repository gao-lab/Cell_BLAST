#! /usr/bin/env Rscript
# by caozj
# Jun 15, 2018
# 11:27:10 AM

suppressPackageStartupMessages({
    library(argparse)
    library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)


parser <- ArgumentParser()
parser$add_argument("-i", "--input", dest = "input", type = "character", required = TRUE)
parser$add_argument("-g", "--genes", dest = "genes", type = "character", default = "seurat_genes")
parser$add_argument("-b", "--batch", dest = "batch", type = "character", required = TRUE)
parser$add_argument("-d", "--dim", dest = "dim", type = "integer", default = 10)
parser$add_argument("-o", "--output", dest = "output", type = "character", required = TRUE)
parser$add_argument("-s", "--seed", dest = "seed", type = "integer", default = NULL)
parser$add_argument("--clean", dest = "clean", type = "character", default = NULL)
args <- parser$parse_args()


# Reading and preprocessing
cat("Reading and preprocessing...\n")
data_obj <- read_dataset(args$input)
if (!is.null(args$clean))
    data_obj <- clean_dataset(data_obj, args$clean)
so <- to_seurat(data_obj, var.genes = args$genes)

unique_batches <- unique(so@meta.data[[args$batch]])
print(unique_batches)
so <- NormalizeData(so)
start_time <- proc.time()
so <- ScaleData(so)


# CCA alignment
cat("Computing CCA...\n")
if (length(unique_batches) > 2) {
    so_list <- list()
    cell.names <- so@cell.names
    for (batch in unique_batches) {
        so_list[[batch]] <-
            SubsetData(so, subset.name = args$batch, accept.value = batch)
    }
    if (!is.null(args$seed)) {
        set.seed(args$seed)
    }
    so <- RunMultiCCA(
        object.list = so_list,
        genes.use = so@var.genes,
        num.ccs = args$dim
    )
    so <- SubsetData(so, cells.use = cell.names)  # Ensure same order
} else {  # length(unique_batches) == 2
    if (!is.null(args$seed)) {
        set.seed(args$seed)
    }
    so <- RunCCA(
        so,
        group1 = unique_batches[1],
        group2 = unique_batches[2],
        group.by = args$batch,
        num.cc = args$dim,
        genes.use = so@var.genes
    )
}


# # Choose proper number of CCs
# DimHeatmap(object = so, reduction.type = "cca",
#            cells.use = 500, dim.use = 1:9,
#            do.balanced = TRUE)
# args$dim <- 20

cat("Performing CCA alignment...\n")
so <- AlignSubspace(so, reduction.type = "cca",
                    grouping.var = args$batch, dims.align = 1:args$dim)
elapsed_time <- proc.time() - start_time

# Save results
cat("Saving result...\n")
write_hybrid_path(t(so@dr$cca.aligned@cell.embeddings),
                  paste(args$output, "latent", sep = "//"))
write_hybrid_path(elapsed_time["elapsed"],
                  paste(args$output, "time", sep = "//"))

cat("Done!\n")
