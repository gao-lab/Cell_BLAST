#! /usr/bin/env Rscript
# by caozj
# Jun 15, 2018
# 11:27:10 AM

source("../packrat/envs/seurat_v3/.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
  library(argparse)
  library(Seurat)
})
source("../Utilities/data.R", chdir = TRUE)


parser <- ArgumentParser()
parser$add_argument("-i", "--input", dest = "input", type = "character", required = TRUE)
parser$add_argument("-g", "--genes", dest = "genes", type = "character", default = "seurat_genes")
parser$add_argument("-b", "--batch", dest = "batch", type = "character", required = TRUE)
parser$add_argument("-d", "--dim", dest = "dim", type = "integer", default = 20)
parser$add_argument("-o", "--output", dest = "output", type = "character", required = TRUE)
parser$add_argument("-s", "--seed", dest = "seed", type = "integer", default = NULL)
parser$add_argument("--clean", dest = "clean", type = "character", default = NULL)
args <- parser$parse_args()


cat("Reading and preprocessing...\n")
data_obj <- read_dataset(args$input)
if (!is.null(args$clean))
  data_obj <- clean_dataset(data_obj, args$clean)
so <- CreateSeuratObject(data_obj@exprs, meta.data = data_obj@obs)
VariableFeatures(so) <- data_obj@uns[[args$genes]]

start_time <- proc.time()
so.list <- SplitObject(so, split.by = args$batch)
for (i in 1:length(so.list)) {
  so.list[[i]] <- NormalizeData(so.list[[i]], verbose = FALSE)
}

cat("Performing integration...\n")
if (!is.null(args$seed)) {
  set.seed(args$seed)
}
so.anchors <- FindIntegrationAnchors(object.list = so.list, dims = 1:args$dim)
so.integrated <- IntegrateData(anchorset = so.anchors, dims = 1:args$dim)
elapsed_time <- proc.time() - start_time

cat("Saving result...\n")
write_hybrid_path(as.matrix(GetAssayData(so.integrated)),
                  paste(args$output, "exprs", sep = "//"))
write_hybrid_path(elapsed_time["elapsed"],
                  paste(args$output, "time", sep = "//"))

cat("Done!\n")
