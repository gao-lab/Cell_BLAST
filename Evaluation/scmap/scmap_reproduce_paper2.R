#! /usr/bin/env Rscript
# by caozj
# Apr 21, 2018
# 3:39:26 PM

# This script tries to reproduce results in the paper by exactly
# following the vignette

suppressPackageStartupMessages({
    library(argparse)
    library(rhdf5)
    library(SingleCellExperiment)
    library(scmap)
})


parser <- ArgumentParser()
parser$add_argument("-r", "--ref", dest = "ref",
                    type = "character", required = TRUE)
parser$add_argument("-q", "--query", dest = "query",
                    type = "character", required = TRUE)
parser$add_argument("-o", "--output", dest = "output",
                    type = "character", required = TRUE)
parser$add_argument("-s", "--seed", dest = "seed",
                    type = "character", default = NULL)
args <- parser$parse_args()


#===============================================================================
#
#  Process reference
#
#===============================================================================
cat("Reading reference data...\n")
ref <- readRDS(args$ref)
# tryCatch({
#     logcounts(ref) <- log2(counts(ref) + 1)
# }, error = function(e) {
#     logcounts(ref) <- log2(normcounts(ref) + 1)
# })
ref <- selectFeatures(ref, suppress_plot = TRUE)


#===============================================================================
#
#  Process query
#
#===============================================================================
cat("Reading query data...\n")
query <- readRDS(args$query)
# tryCatch({
#     logcounts(query) <- log2(counts(query) + 1)
# }, error = function(e) {
#     logcounts(query) <- log2(normcounts(query) + 1)
# })


#===============================================================================
#
#  scmap-cluster
#
#===============================================================================
cat("Performing scmap-cluster...\n")
ref <- indexCluster(ref)
scmapCluster_results <- scmapCluster(
    projection = query,
    index_list = list(
        ref = metadata(ref)$scmap_cluster_index
    )
)
cat(sprintf(
    "\tCoverage = %f\n",
    sum(scmapCluster_results$scmap_cluster_labs[, 1] != "unassigned") /
        nrow(scmapCluster_results$scmap_cluster_labs)
))



#===============================================================================
#
#  scmap-cell
#
#===============================================================================
cat("Performing scmap-cell...\n")
if (!is.null(args$seed)) {
    set.seed(args$seed)
}
ref <- indexCell(ref)
scmapCell_results <- scmapCell(
    projection = query,
    index_list = list(
        ref = metadata(ref)$scmap_cell_index
    )
)
scmapCell_clusters <- scmapCell2Cluster(
    scmapCell_results = scmapCell_results,
    cluster_list = list(
        as.character(colData(ref)$cell_type1)
    )
)
cat(sprintf(
    "\tCoverage = %f\n",
    sum(scmapCell_clusters$scmap_cluster_labs[, 1] != "unassigned") /
        nrow(scmapCell_clusters$scmap_cluster_labs)
))



#===============================================================================
#
#  Save results
#
#===============================================================================
cat("Saving results...\n")
if (! dir.exists(dirname(args$output))) {
    cat("Creating directory...\n")
    dir.create(dirname(args$output), recursive = TRUE)
}
if (file.exists(args$output)) {
    cat("Removing previous file...\n")
    stopifnot(file.remove(args$output))
}
stopifnot(h5createFile(args$output))
h5write(scmapCluster_results$scmap_cluster_labs[, 1],
        args$output, "scmap_cluster")
h5write(scmapCell_clusters$scmap_cluster_labs[, 1],
        args$output, "scmap_cell")


message("Done!")
