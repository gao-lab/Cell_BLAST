#! /usr/bin/env Rscript
# by caozj
# Apr 21, 2018
# 3:39:26 PM

# This script uses scmap for cell annotation based on reference dataset

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(argparse)
    library(SingleCellExperiment)
    library(scmap)
})
source("../Utilities/data.R")


parser <- ArgumentParser()
parser$add_argument("-r", "--refs", dest = "refs", nargs = "+", type = "character")
parser$add_argument("-n", "--ref-names", dest = "ref_names", nargs = "*", type = "character")
parser$add_argument("-g", "--genes", dest = "genes", nargs = "*", type = "character", default=NULL)
parser$add_argument("-c", "--cluster-col", dest = "cluster_col", nargs = "+", default = "cell_type1")
parser$add_argument("-q", "--query", dest = "query", type = "character", required = TRUE)
parser$add_argument("-o", "--output", dest = "output", type = "character", required = TRUE)
parser$add_argument("-s", "--seed", dest = "seed", type = "character", default = NULL)
parser$add_argument("--n-neighbors", dest = "n_neighbors", type = "double", default = 10)
parser$add_argument("--threshold", dest = "threshold", nargs = "+", default = 0.5)
parser$add_argument("--self", dest = "self", default = FALSE, action = "store_true")
parser$add_argument("--do-scmap-cluster", dest = "do_scmap_cluster", default = FALSE, action = "store_true")
parser$add_argument("--shuffle-genes", dest = "shuffle_genes", default = FALSE, action = "store_true")
parser$add_argument("--subsample-ref", dest = "subsample_ref", type = "integer", default = NULL)
parser$add_argument("--clean", dest = "clean", type = "character", default = NULL)
args <- parser$parse_args()
if (!is.null(args$seed)) {
    set.seed(args$seed)
}


#===============================================================================
#
#  Aux functions
#
#===============================================================================
compute_coverage <- function(pred) {
    assigned_mask <- pred != "unassigned"
    sum(assigned_mask) / length(assigned_mask)
}

print_coverage <- function(coverage_list) {
    for (item in names(coverage_list)) {
        cat(sprintf("Coverage for %s = %f\n", item, coverage_list[[item]]))
    }
}


#===============================================================================
#
#  Process reference
#
#===============================================================================
cat("Reading reference data...\n")
ref_sces <- list()
for (i in 1:length(args$refs)) {
    cat(sprintf("Processing \"%s\"...\n", args$refs[i]))
    ref_dataset <- read_dataset(args$refs[i])
    ref_dataset <- normalize(ref_dataset)
    if (!is.null(args$clean))
        ref_dataset <- clean_dataset(ref_dataset, args$clean)
    if (!is.null(args$subsample_ref)) {
        subsample_idx <- sample(
            dim(ref_dataset)[2], args$subsample_ref, replace = FALSE)
        ref_dataset <- ref_dataset[, subsample_idx]
    }
    if (args$shuffle_genes) {
        subsample_idx <- sample(
            rownames(ref_dataset), nrow(ref_dataset), replace = FALSE)
        ref_dataset <- ref_dataset[subsample_idx, ]
    }
    if (!is.null(args$genes)) {
        if (length(args$genes) == 1)
            this_gene <- args$genes
        else
            this_gene <- args$genes[i]
        if (grepl("//", this_gene))
            used_genes <- read_hybrid_path(this_gene)
        else
            used_genes <- ref_dataset@uns[[this_gene]]
        ref_dataset <- ref_dataset[rownames(ref_dataset) %in% used_genes, ]
    }
    ref_sce <- SingleCellExperiment(
        assays = list(normcounts = as.matrix(ref_dataset@exprs)),
        colData = ref_dataset@obs
    )
    logcounts(ref_sce) <- log1p(normcounts(ref_sce))
    rowData(ref_sce)$feature_symbol <- rownames(ref_sce)
    print(dim(ref_sce))
    if (!is.null(args$genes)) {
        rowData(ref_sce)$scmap_features <-
            rowData(ref_sce)$feature_symbol %in% used_genes
    } else {
        ref_sce <- selectFeatures(ref_sce, suppress_plot = TRUE)
    }
    if (length(args$cluster_col) == 1)
        this_col <- args$cluster_col
    else
        this_col <- args$cluster_col[i]
    colData(ref_sce)[["cell_type1"]] <- colData(ref_sce)[[this_col]]
    if (args$do_scmap_cluster)
        ref_sce <- indexCluster(ref_sce)
    ref_sce <- indexCell(ref_sce)
    if (is.null(args$ref_names))
        ref_sces[[i]] <- ref_sce
    else
        ref_sces[[args$ref_names[i]]] <- ref_sce
}


#===============================================================================
#
#  Process query
#
#===============================================================================
if (!is.null(args$query)) {
    cat("Reading query data...\n")
    query <- read_dataset(args$query)
    query <- normalize(query)
    if (!is.null(args$clean))
        query <- clean_dataset(query, args$clean)
    query_sce <- SingleCellExperiment(
        assays = list(normcounts = as.matrix(query@exprs)),
        colData = query@obs
    )
    logcounts(query_sce) <- log1p(normcounts(query_sce))
    rowData(query_sce)$feature_symbol <- rownames(query_sce)
} else {
    query_sce <- Reduce(cbind, ref_sces)
}


#===============================================================================
#
#  scmap-cluster
#
#===============================================================================
if (args$do_scmap_cluster) {
    cat("Performing scmap-cluster...\n")
    start_time <- Sys.time()
    scmapCluster_results <- scmapCluster(
        projection = query_sce,
        index_list = lapply(
            ref_sces, function(sce) metadata(sce)$scmap_cluster_index
        )
    )
    scmap_cluster_ms_per_cell <- 1000 * as.numeric(difftime(
        Sys.time(), start_time, units = "secs"
    )) / ncol(query_sce)
    print(sprintf("Time: %f ms/cell", scmap_cluster_ms_per_cell))
    # print_coverage(
    #     lapply(cbind(
    #         scmapCluster_results$scmap_cluster_labs,
    #         data.frame(combined = scmapCluster_results$combined_labs)
    #     ), compute_coverage)
    # )
}


#===============================================================================
#
#  scmap-cell
#
#===============================================================================
cat("Performing scmap-cell...\n")
start_time <- Sys.time()
if (args$n_neighbors < 1) {
    stopifnot(length(ref_sces == 1))
    args$n_neighbors <- round(args$n_neighbors * dim(ref_sces[[1]])[2])
    if (args$self) args$n_neighbors <- args$n_neighbors + 1
}  # Specifies a fraction w.r.t reference size, only makes sense with one reference
scmapCell_results <- scmapCell(
    projection = query_sce,
    index_list = lapply(
        ref_sces, function(sce) metadata(sce)$scmap_cell_index
    ), w = as.integer(args$n_neighbors)
)
scmap_cell_ms_per_cell <- NULL
scmapCell_clusters <- list()
for (threshold in args$threshold) {
    scmapCell_clusters[[as.character(threshold)]] <- scmapCell2Cluster(
        scmapCell_results = scmapCell_results,
        cluster_list = lapply(
            ref_sces, function(sce) as.character(colData(sce)[["cell_type1"]])
        ), threshold = as.numeric(threshold)
    )
    if (is.null(scmap_cell_ms_per_cell)) {
        scmap_cell_ms_per_cell <- 1000 * as.numeric(difftime(
            Sys.time(), start_time, units = "secs"
        )) / ncol(query_sce)
    }
}
print(sprintf("Time: %f ms/cell", scmap_cell_ms_per_cell))


#===============================================================================
#
#  Save results
#
#===============================================================================
cat("Saving results...\n")
if (! dir.exists(dirname(args$output))) {
    dir.create(dirname(args$output), recursive = TRUE)
}
if (file.exists(args$output)) {
    cat("Removing previous file...\n")
    stopifnot(file.remove(args$output))
}

f <- H5Fcreate(args$output)

if (args$do_scmap_cluster) {
    g <- H5Gcreate(f, "scmap_cluster")
    list_to_group(as.list(as.data.frame(
        scmapCluster_results$scmap_cluster_labs
    )), H5Gcreate(g, "labs"))
    list_to_group(as.list(as.data.frame(
        scmapCluster_results$scmap_cluster_siml
    )), H5Gcreate(g, "siml"))
    h5write(scmapCluster_results$combined_labs, g, "combined_labs")
    h5write(scmap_cluster_ms_per_cell, g, "time")
}

g <- H5Gcreate(f, "scmap_cell")
for (threshold in names(scmapCell_clusters)) {
    subg <- H5Gcreate(g, threshold)
    list_to_group(as.list(as.data.frame(
        scmapCell_clusters[[threshold]]$scmap_cluster_labs
    )), H5Gcreate(subg, "labs"))
    list_to_group(as.list(as.data.frame(
        scmapCell_clusters[[threshold]]$scmap_cluster_siml
    )), H5Gcreate(subg, "siml"))
    h5write(scmapCell_clusters[[threshold]]$combined_labs,
            subg, "combined_labs")
}
list_to_group(scmapCell_results, H5Gcreate(g, "nn"))
h5write(scmap_cell_ms_per_cell, g, "time")

# For compatibility
g <- H5Gcreate(f, "prediction")
for (threshold in names(scmapCell_clusters))
    h5write(scmapCell_clusters[[threshold]]$combined_labs,
            g, threshold)
h5write(scmap_cell_ms_per_cell, f, "time")
h5write(sum(sapply(ref_sces, ncol)), f, "ref_size")

H5Fclose(f)

message("Done!")
