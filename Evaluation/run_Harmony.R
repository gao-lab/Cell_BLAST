#! /usr/bin/env Rscript
# by caozj
# Apr 30, 2020
# 11:42:10 PM

source("../packrat/envs/seurat_v3/.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(argparse)
    library(Seurat)
    library(harmony)
})
source("../Utilities/data.R", chdir = TRUE)


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
so <- CreateSeuratObject(data_obj@exprs, meta.data = data_obj@obs)
VariableFeatures(so) <- data_obj@uns[[args$genes]]

unique_batches <- unique(so@meta.data[[args$batch]])
print(unique_batches)
so <- NormalizeData(so)
start_time <- proc.time()
so <- ScaleData(so)

if (!is.null(args$seed)) {
    set.seed(args$seed)
}
so <- RunPCA(so, features = VariableFeatures(so), npcs = args$dim, verbose = FALSE)
so <- RunHarmony(so, group.by.vars = args$batch)

elapsed_time <- proc.time() - start_time

# Save results
cat("Saving result...\n")
write_hybrid_path(t(Embeddings(so, reduction="harmony")),
                  paste(args$output, "latent", sep = "//"))
write_hybrid_path(elapsed_time["elapsed"],
                  paste(args$output, "time", sep = "//"))

cat("Done!\n")
