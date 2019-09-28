#! /usr/bin/env Rscript
# by caozj
# May 13, 2018
# 9:18:48 AM

suppressPackageStartupMessages({
    library(Matrix)
    library(dplyr)
    library(rhdf5)
})


#===============================================================================
#
#  Utility functions
#
#===============================================================================
read_clean <- function(data) {  # Convert arrays
    if (length(dim(data)) == 1) {
        data <- as.vector(data)
    } else if (length(dim(data)) == 2) {
        data <- as.matrix(data)
    }
    data
}


write_clean <- function(data) {
    if (is.factor(data))
        data <- as.character(data)
    if (is.character(data)) {
        data[is.na(data)] <- "NA"
    } else if (is.numeric(data)) {
        data[is.na(data)] <- 0
    }
    data
}


list_from_group <- function(group) {
    l <- list()
    content <- h5ls(group, recursive = FALSE)
    for (i in rownames(content)) {
        name <- content[i, "name"]
        type <- content[i, "otype"]
        if (type == "H5I_DATASET") {
            result <- read_clean(h5read(group, name))
            l[[name]] <- result
        } else {
            stopifnot(type == "H5I_GROUP")
            l[[name]] <- list_from_group(H5Gopen(group, name))
        }
    }
    l
}


list_to_group <- function(l, group) {
    force(group)
    for (name in names(l)) {
        content <- l[[name]]
        if (is.list(content)) {  # Recursive group
            list_to_group(content, H5Gcreate(group, name))
        } else {  # Write dataset
            h5write(write_clean(content), group, name)
        }
    }
}


read_expr_mat <- function(file) {
    exprs_handle <- H5Oopen(file, "exprs")
    if (H5Iget_type(exprs_handle) == "H5I_GROUP") {  # Sparse
        mat <- new(
            "dgCMatrix",
            x = as.double(read_clean(h5read(exprs_handle, "data"))),
            i = read_clean(h5read(exprs_handle, "indices")),
            p = read_clean(h5read(exprs_handle, "indptr")),
            Dim = rev(read_clean(h5read(exprs_handle, "shape")))
        )
    } else if (H5Iget_type(exprs_handle) == "H5I_DATASET") {  # Dense
        mat <- read_clean(H5Dread(exprs_handle))
    }
    mat
}


write_expr_mat <- function(mat, file) {
    message(sprintf("Saving to expression matrix...\n"))
    if (is(mat, "dMatrix")) {  # Sparse
        g <- H5Gcreate(file, "exprs")
        mat <- as(mat, "dgCMatrix")
        suppressWarnings({  # Suppress chunk size warning
            h5write(mat@x, g, "data")
            h5write(mat@i, g, "indices")
            h5write(mat@p, g, "indptr")
            h5write(rev(mat@Dim), g, "shape")
        })
    } else if (is.matrix(mat)) {  # Dense
        stopifnot(h5createDataset(
            file, "exprs", dims = dim(mat),
            chunk = c(min(200, dim(mat)[1]), min(200, dim(mat)[2]))
        ))
        h5writeDataset(mat, file, "exprs")
    } else {
        stop("Unsupported matrix type!")
    }
}


recursive_create_group <- function(loc, path) {
    path_split <- strsplit(path, "/")[[1]]
    if (path_split[1] == "") {  # Path starts with "/"
        path_split <- path_split[-1]
    }
    for (item in path_split) {
        if (item %in% h5ls(loc, recursive = FALSE)$name) {
            loc <- H5Gopen(loc, item)
        } else {
            loc <- H5Gcreate(loc, item)
        }
    }
    loc
}


check_hybrid_path <- function(path) {
    path_split <- strsplit(path, "//")[[1]]
    filename <- path_split[[1]]
    h5_path <- path_split[[2]]
    file_content <- h5ls(filename)
    file_content <- paste(file_content$group, file_content$name, sep = "/")
    file_content <- gsub("^/+", "", file_content)
    h5_path %in% file_content
}


read_hybrid_path <- function(path) {
    path_split <- strsplit(path, "//")[[1]]
    filename <- path_split[[1]]
    h5_path <- path_split[[2]]
    handle <- H5Oopen(H5Fopen(filename, flags="H5F_ACC_RDONLY"), h5_path)
    if (H5Iget_type(handle) == "H5I_GROUP") {
        result <- list_from_group(handle)
    } else {  # "H5I_DATASET"
        result <- H5Dread(handle)
        if (length(dim(result)) == 1) {
            result <- as.vector(result)
        } else if (length(dim(result)) == 2) {
            result <- as.matrix(result)
        }
    }
    H5close()
    result
}


write_hybrid_path <- function(obj, path) {
    path_split <- strsplit(path, "//")[[1]]
    file_name <- path_split[[1]]
    h5_path <- path_split[[2]]
    if (! dir.exists(dirname(file_name)))
        dir.create(dirname(file_name), recursive = TRUE)
    if (! file.exists(file_name)) {
        file <- H5Fcreate(file_name)
    } else {
        file <- H5Fopen(file_name)
    }
    h5_dirname <- dirname(h5_path)
    if (is.list(obj)) {
        list_to_group(obj, recursive_create_group(file, h5_path))
    } else {
        if (dirname(h5_path) != ".")
            recursive_create_group(file, dirname(h5_path))
        h5write(write_clean(obj), file, h5_path)
    }
    H5close()
}


assign_CL <- function(meta_df, cell_ontology){
    meta_df$rownm <- rownames(meta_df)
    meta_df <- merge(meta_df, cell_ontology, by = "cell_type1", all.x = TRUE)
    rownames(meta_df) <- meta_df$rownm
    meta_df$rownm <- NULL
    meta_df
}


construct_dataset <- function(
    save_dir, expr_mat, cell_meta, datasets_meta = NULL, cell_ontology = NULL,
    gene_meta = NULL, gene_list = NULL, sparse=TRUE, ...
) {
    if (! dir.exists(save_dir)) {
      message("Creating directory...\n")
      dir.create(save_dir, recursive = TRUE)
    }

    if (sparse) {
        expr_mat <- Matrix(expr_mat, sparse = TRUE)
    } else {
        expr_mat <- as.matrix(expr_mat)
    }

    # Assign_CL
    if (!is.null(cell_ontology)) {
        cell_meta <- assign_CL(cell_meta, cell_ontology)
    }
    cell_meta <- cell_meta[colnames(expr_mat), ,drop = FALSE]

    # Clean up not expressed
    # TODO: probably we should remove this
    # message("Filtering genes...")
    # expr_mat <- expr_mat[Matrix::rowSums(expr_mat) > 0, ]

    # Add dataset metadata
    # What if save_dir changes?
    if (!is.null(datasets_meta)){
        dataset_name <- strsplit(save_dir, "/")[[1]][3]
        cell_meta$dataset_name = dataset_name
        cell_meta$organism = datasets_meta[dataset_name, ]$organism
        cell_meta$organ = datasets_meta[dataset_name, ]$organ
        cell_meta$platform = datasets_meta[dataset_name, ]$platform
    }

    # Construct dataset
    message("Constructing dataset...")
    if (is.null(gene_meta)) {
        gene_meta <- data.frame(row.names = rownames(expr_mat))
    } else {
        gene_meta <- gene_meta[rownames(expr_mat), , drop = FALSE]
    }

    dataset <- new("ExprDataSet",
        exprs = expr_mat,
        obs = cell_meta,
        var = gene_meta,
        uns = if (is.null(gene_list)) list() else gene_list
    )

    scmap_genes <- tryCatch(
        select_genes_scmap(dataset, save_dir, ... = ...),
        error = function(err) {
            print(paste("MY_ERROR:  ",err))
            message("Without selecting scmap genes")
            scmap_genes <- NULL
        }
    )
    seurat_genes <- tryCatch(
        select_genes_seurat(dataset, save_dir, ... = ...),
        error = function(err) {
            print(paste("MY_ERROR:  ",err))
            message("Without selecting seurat genes")
            message(geterrmessage())
            seurat_genes <- NULL
        }
    )
    expressed_genes <- rownames(expr_mat)[Matrix::rowSums(expr_mat > 1) > 5]

    dataset@uns$expressed_genes <- union(
        dataset@uns$expressed_genes, expressed_genes)
    dataset@uns$scmap_genes <- union(
        dataset@uns$scmap_genes, scmap_genes)
    dataset@uns$seurat_genes <- union(
        dataset@uns$seurat_genes, seurat_genes)

    message("Saving data...")
    write_dataset(dataset, file.path(save_dir, "data.h5"))
}


select_genes_scmap <- function(
    dataset, save_dir, grouping = NULL, min_group_frac = 0.5,
    n_features = 500, ...
) {
    suppressPackageStartupMessages({
        require(scmap)
        require(SingleCellExperiment)
    })
    dataset <- normalize(dataset)
    dataset_list <- list()
    if (is.null(grouping)) {
        dataset_list[["1"]] <- dataset
    } else {
        grouping <- dataset@obs[, grouping]
        for (group in unique(grouping)) {
            dataset_list[[as.character(group)]] <- dataset[, grouping == group]
        }
    }
    selected_genes <- character()
    for (dataset_name in names(dataset_list)) {
        dataset <- dataset_list[[dataset_name]]
        sce <- SingleCellExperiment(
            assays = list(normcounts = as.matrix(dataset@exprs)),
            colData = dataset@obs
        )
        logcounts(sce) <- log1p(normcounts(sce))
        rowData(sce)$feature_symbol <- rownames(sce)
        if (length(dataset_list) > 1) {
            pdf(file.path(save_dir, sprintf(
                "scmap_genes_%s.pdf", gsub("/", "_", dataset_name)
            )))
        } else {
            pdf(file.path(save_dir, "scmap_genes.pdf"))
        }
        tryCatch({
            sce <- selectFeatures(
                sce, n_features = n_features, suppress_plot = FALSE
            )
            selected_genes <- c(
                selected_genes,
                rownames(sce)[rowData(sce)$scmap_features]
            )
        }, error = function(e) {})
        dev.off()
    }
    selected_genes <- table(selected_genes)
    selected_genes <- names(selected_genes)[
        selected_genes >= min_group_frac * length(dataset_list)]
    return(sort(selected_genes))
}


select_genes_seurat <- function(
    dataset, save_dir, grouping = NULL, min_group_frac = 0.5,
    x_low = 0.1, x_high = 8, y_low = 1, y_high = NULL,
    binning = "equal_frequency", ...
) {
    suppressPackageStartupMessages({
        require(Seurat)
    })
    if (is.null(x_high))
        x_high <- Inf
    if (is.null(x_low))
        x_low <- -Inf
    if (is.null(y_high))
        y_high <- Inf
    if (is.null(y_low))
        y_low <- -Inf
    dataset_list <- list()
    if (is.null(grouping)) {
        dataset_list[["1"]] <- dataset
    } else {
        grouping <- dataset@obs[, grouping]
        for (group in unique(grouping)) {
            dataset_list[[as.character(group)]] <- dataset[, grouping == group]
        }
    }
    var.genes <- character()
    for (dataset_name in names(dataset_list)) {
        dataset <- dataset_list[[dataset_name]]
        so <- to_seurat(dataset)
        so <- NormalizeData(
            object = so, normalization.method = "LogNormalize",
            scale.factor = 10000
        )
        if (length(dataset_list) > 1) {
            pdf(file.path(save_dir, sprintf(
                "seurat_genes_%s.pdf", gsub("/", "_", dataset_name)
            )))
        } else {
            pdf(file.path(save_dir, "seurat_genes.pdf"))
        }
        tryCatch({
            so <- FindVariableGenes(
                object = so, mean.function = ExpMean, dispersion.function = LogVMR,
                binning.method = binning,
                x.low.cutoff = x_low, x.high.cutoff = x_high,
                y.cutoff = y_low, y.high.cutoff = y_high
            )
            message(sprintf(
                "Number of variable genes for %s: %d\n",
                dataset_name, length(so@var.genes)
            ))
            var.genes <- c(var.genes, so@var.genes)
        }, error = function(e) {})
        dev.off()
    }
    var.genes <- table(var.genes)
    var.genes <- names(var.genes)[
        var.genes >= min_group_frac * length(dataset_list)]
    message(sprintf("Number of variable genes: %d\n", length(var.genes)))
    return(sort(var.genes))
}


#===============================================================================
#
#  ExprDataSet
#
#===============================================================================
ExprDataSet <- setClass(
    "ExprDataSet", slots = representation(
        exprs = "ANY",  # either dense or sparse
        obs = "data.frame",
        var = "data.frame",
        uns = "list"
    ), prototype = list(
        exprs = matrix(nrow = 0, ncol = 0),
        obs = data.frame(),
        var = data.frame(),
        uns = list()
    ), validity = function(object) {
        nrow(object@exprs) == nrow(object@var) &&
        ncol(object@exprs) == nrow(object@obs) &&
        all(rownames(object@exprs) == rownames(object@var)) &&
        all(colnames(object@exprs) == rownames(object@obs))
    }
)

read_dataset <- function(filename) {

    f <- H5Fopen(filename, flags="H5F_ACC_RDONLY")

    obs_names <- as.vector(H5Dread(H5Dopen(f, "obs_names")))
    var_names <- as.vector(H5Dread(H5Dopen(f, "var_names")))
    obs <- as.data.frame(list_from_group(H5Gopen(f, "obs")))
    var <- as.data.frame(list_from_group(H5Gopen(f, "var")))
    if (nrow(obs) == 0)
        obs <- data.frame(row.names = obs_names)
    else
        rownames(obs) <- obs_names
    if (nrow(var) == 0)
        var <- data.frame(row.names = var_names)
    else
        rownames(var) <- var_names
    uns <- list_from_group(H5Gopen(f, "uns"))

    exprs <- read_expr_mat(f)
    colnames(exprs) <- obs_names
    rownames(exprs) <- var_names

    H5close()

    new("ExprDataSet", exprs = exprs, obs = obs, var = var, uns = uns)
}

setMethod(
    "rownames",
    signature = signature(x = "ExprDataSet"),
    definition = function(x) {
        rownames(x@exprs)
    }
)

setMethod(
    "colnames",
    signature = signature(x = "ExprDataSet"),
    definition = function(x) {
        colnames(x@exprs)
    }
)

setMethod(
    "dim",
    signature = signature(x = "ExprDataSet"),
    definition = function(x) {
        dim(x@exprs)
    }
)

setMethod(
    "nrow",
    signature = signature(x = "ExprDataSet"),
    definition = function(x) {
        nrow(x@exprs)
    }
)

setMethod(
    "ncol",
    signature = signature(x = "ExprDataSet"),
    definition = function(x) {
        ncol(x@exprs)
    }
)

setMethod(
    "[", signature = signature(x = "ExprDataSet"),
    definition = function(x, i, j, silent = FALSE) {
        if (missing(i))
            i <- 1:nrow(x)
        if (missing(j))
            j <- 1:ncol(x)

        # Subset var
        if (is.character(i)) {
            new_var_names <- setdiff(i, rownames(x))
            all_var_names <- union(rownames(x), new_var_names)
            if (length(new_var_names) > 0 && ! silent) {
                message(sprintf("[Warning] %d out of %d genes not found, set to zero",
                                length(new_var_names), length(i)))
                message(paste(new_var_names, collapse = ","))
            }
            if (is(x@exprs, "dMatrix")) {
                exprs <- rbind(x@exprs, Matrix(
                    0, nrow = length(new_var_names), ncol = ncol(x@exprs),
                    dimnames = list(new_var_names, colnames(x@exprs)),
                    sparse = TRUE
                ))
            } else {
                exprs <- rbind(x@exprs, matrix(
                    0, nrow = length(new_var_names), ncol = ncol(x@exprs),
                    dimnames = list(new_var_names, colnames(x@exprs)),
                    sparse = TRUE
                ))
            }
        } else {
            exprs <- x@exprs
        }
        if (is.character(i) || is.integer(i) || is.logical(i)) {
            exprs <- exprs[i, , drop = FALSE]
            var <- x@var[i, , drop = FALSE]
            if (is.character(i))
                rownames(var) <- i
        } else {
            stop("Unsupported var index!")
        }

        # Subset obs
        if (is.character(j))
            stopifnot(all(j %in% colnames(x)))
        if (is.character(j) || is.integer(j) || is.logical(j)) {
            exprs <- exprs[, j, drop = FALSE]
            obs <- x@obs[j, , drop = FALSE]
            if (is.character(j))
                rownames(obs) <- j
        } else {
            stop("Unsupported obs index!")
        }

        new("ExprDataSet", exprs = exprs, obs = obs, var = var, uns = x@uns)
    }
)

setGeneric("to_seurat", function(object, ...) {
    stop("Calling generic function `to_seurat`!")
})
setMethod(
    "to_seurat",
    signature = signature(object = "ExprDataSet"),
    definition = function(object, var.genes = NULL) {
        suppressPackageStartupMessages({
            require(Seurat)
        })
        so <- CreateSeuratObject(raw.data = object@exprs,
                                 meta.data = object@obs)
        if (!is.null(var.genes))
            so@var.genes <- object@uns[[var.genes]]
        so
    }
)

setGeneric("normalize", function(object, ...) {
    stop("Calling generic function `normalize`!")
})
setMethod(
    "normalize",
    signature = signature(object = "dMatrix"),
    definition = function(object, target = 10000) {
        object <- as(object, "dgCMatrix")
        object@x <- target * object@x /
            rep.int(Matrix::colSums(object), diff(object@p))
        object
    }
)
setMethod(
    "normalize",
    signature = signature(object = "matrix"),
    definition = function(object, target = 10000) {
        object <- t(object)
        object <- object / (rowSums(object) / target)
        t(object)
    }
)
setMethod(
    "normalize",
    signature = signature(object = "ExprDataSet"),
    definition = function(object, target = 10000) {
        object@exprs <- normalize(object@exprs, target = target)
        rownames(object@exprs) <- rownames(object@var)
        colnames(object@exprs) <- rownames(object@obs)
        object
    }
)

setGeneric("write_dataset", function(object, ...) {
    stop("Calling generic function `write_dataset`!")
})
setMethod(
    "write_dataset",
    signature = signature(object = "ExprDataSet"),
    definition = function(object, file) {
        if (! dir.exists(dirname(file))) {
            message("Creating directory...\n")
            dir.create(dirname(file), recursive = TRUE)
        }
        if (file.exists(file)) {
            message("Removing previous file...\n")
            file.remove(file)
        }

        # Save to hdf5 file
        file <- H5Fcreate(file)
        write_expr_mat(object@exprs, file)

        h5write(colnames(object@exprs), file, "obs_names")
        h5write(rownames(object@exprs), file, "var_names")

        list_to_group(as.list(object@obs), H5Gcreate(file, "obs"))
        list_to_group(as.list(object@var), H5Gcreate(file, "var"))
        list_to_group(object@uns, H5Gcreate(file, "uns"))
        H5close()
    }
)


clean_dataset <- function(dataset, obs_col) {
    obs_col <- as.character(dataset@obs[, obs_col])
    mask <- obs_col %in% list("", "na", "NA", "nan", "NaN") | is.na(obs_col)
    message(sprintf("Cleaning removed %d cells.", sum(mask)))
    dataset[, !mask]
}
