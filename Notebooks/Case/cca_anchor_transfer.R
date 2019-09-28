#! /usr/bin/env Rscript
# by caozj
# Jul 3, 2019
# 2:59:59 PM

source("../../packrat/envs/seurat_v3/.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(argparse)
    library(Seurat)
})
source("../../Utilities/data.R", chdir = TRUE)

parser <- ArgumentParser()
parser$add_argument("-r", "--ref", dest = "ref", type = "character", required = TRUE)
parser$add_argument("-q", "--query", dest = "query", type = "character", required = TRUE)
parser$add_argument("-g", "--genes", dest = "genes", type = "character", required = TRUE)
parser$add_argument("-a", "--annotate", dest = "annotate", type = "character", nargs = "+")
parser$add_argument("-d", "--dim", dest = "dim", type = "integer", default = 20)
parser$add_argument("-o", "--output", dest = "output", type = "character", required = TRUE)
parser$add_argument("-s", "--seed", dest = "seed", type = "character", default = NULL)
parser$add_argument("--clean", dest = "clean", type = "character", default = NULL)
args <- parser$parse_args()

cat("Reading and preprocessing...\n")
ref_obj <- read_dataset(args$ref)
if (!is.null(args$clean))
    ref_obj <- clean_dataset(ref_obj, args$clean)
ref <- CreateSeuratObject(ref_obj@exprs, meta.data = ref_obj@obs)
VariableFeatures(ref) <- ref_obj@uns[[args$genes]]

query_obj <- read_dataset(args$query)
if (!is.null(args$clean))
    query_obj <- clean_dataset(query_obj, args$clean)
query <- CreateSeuratObject(query_obj@exprs, meta.data = query_obj@obs)

cat("Performing integration...\n")
transfer.anchors <- FindTransferAnchors(reference = ref, query = query, dims = 1:args$dim)
if (length(args$annotate) > 1) {
    predictions <- TransferData(
        anchorset = transfer.anchors,
        refdata = t(as.matrix(ref@meta.data[, args$annotate])),
        dims = 1:args$dim
    )
} else {
    predictions <- TransferData(
        anchorset = transfer.anchors,
        refdata = ref@meta.data[[args$annotate]],
        dims = 1:args$dim
    )
}

cat("Saving result...\n")
if (file.exists(args$output))
    file.remove(args$output)
if (length(args$annotate) > 1) {
    write_hybrid_path(as.matrix(predictions@data), paste(args$output, "prediction", sep = "//"))
} else {
    print(table(predictions$predicted.id))
    write_hybrid_path(predictions$predicted.id, paste(args$output, "prediction", sep = "//"))
}

cat("Done!\n")
