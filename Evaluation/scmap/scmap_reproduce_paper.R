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
parser$add_argument("--exprs", dest = "exprs",
                    type = "character", default = "norm.data")
parser$add_argument("--annotation", dest = "annotation",
                    type = "character", default = "meta/cell_type1")
args <- parser$parse_args()


#===============================================================================
#
#  Process reference
#
#===============================================================================
cat("Reading reference data...\n")
ref <- h5read(args$ref, args$exprs)
gene_names <- as.vector(h5read(args$ref, "meta/gene_names"))
cell_names <- as.vector(h5read(args$ref, "meta/cell_names"))
rownames(ref) <- gene_names
colnames(ref) <- cell_names
cdata <- data.frame(
    cell_type1 = h5read(args$ref, args$annotation),
    row.names = cell_names,
    stringsAsFactors = FALSE
)

ref <- SingleCellExperiment(
    assays = list(normcounts = ref), colData = cdata
)
logcounts(ref) <- log2(normcounts(ref) + 1)
rowData(ref)$feature_symbol <- rownames(ref)
isSpike(ref, "ERCC") <- grepl("^ERCC-", rownames(ref))
ref <- ref[!duplicated(rownames(ref)), ]
ref <- selectFeatures(ref, suppress_plot = TRUE)


#===============================================================================
#
#  Process query
#
#===============================================================================
cat("Reading query data...\n")
query <- h5read(args$query, args$exprs)
gene_names <- as.vector(h5read(args$query, "meta/gene_names"))
cell_names <- as.vector(h5read(args$query, "meta/cell_names"))
rownames(query) <- gene_names
colnames(query) <- cell_names
cdata <- data.frame(
    row.names = cell_names,
    stringsAsFactors = FALSE
)

query <- SingleCellExperiment(
    assays = list(normcounts = query), colData = cdata
)
logcounts(query) <- log2(normcounts(query) + 1)
rowData(query)$feature_symbol <- rownames(query)
isSpike(query, "ERCC") <- grepl("^ERCC-", rownames(query))
query <- query[!duplicated(rownames(query)), ]


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
