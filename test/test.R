#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(unittest)
    library(Matrix)
})
source("../Utilities/data.R")


# Test hybrid path
suppressMessages({
    m1 <- matrix(c(
        2, 1, NA,
        3, 0, 0,
        NA, 4, 5
    ), ncol = 3, byrow = TRUE)  # Numeric
    v2 <- c("a", "s", "d", NA)  # Character
    l3 <- list(m1 = m1, v2 = v2)  # List
    s4 <- "asd"
    write_hybrid_path(m1, "./test.h5//a")
    write_hybrid_path(v2, "./test.h5//b/c")
    write_hybrid_path(l3, "./test.h5//b/d/e")
    write_hybrid_path(s4, "./test.h5//f")
    m1_ok <- matrix(c(
        2, 1, 0,
        3, 0, 0,
        0, 4, 5
    ), ncol = 3, byrow = TRUE)
    v2_ok <- c("a", "s", "d", "NA")
    l3_ok <- list(m1 = m1_ok, v2 = v2_ok)
    s4_ok <- "asd"
})
ok_group("Read and write hybrid path", {
    ok(all.equal(m1_ok, read_hybrid_path("./test.h5//a")), "Numeric")
    ok(all.equal(v2_ok, read_hybrid_path("./test.h5//b/c")), "Character")
    ok(all.equal(l3_ok, read_hybrid_path("./test.h5//b/d/e")), "List")
    ok(all.equal(s4_ok, read_hybrid_path("./test.h5//f")), "Scala")
    ok(check_hybrid_path("./test.h5//b/d/e"))
    ok(!check_hybrid_path("./test.h5//b/f/e"))
})


# ExprDataSet
suppressMessages({
    exprs <- m1_ok
    rownames(exprs) <- c("a", "b", "c")
    colnames(exprs) <- c("d", "e", "f")
    var <- data.frame(column = c(1, 2, 3), row.names = rownames(exprs))
    obs <- data.frame(column = c(1, 2, 3), row.names = colnames(exprs))
    uns <- list(item = "test")

    ds <- new("ExprDataSet", exprs = exprs, obs = obs, var = var, uns = uns)
    write_dataset(ds, "test.h5")

    exprs_name_ok <- matrix(c(
        2, 0,
        0, 0,
        0, 5
    ), ncol = 2, byrow = TRUE)
    rownames(exprs_name_ok) <- c("a", "g", "c")
    colnames(exprs_name_ok) <- c("d", "f")
    var_name_ok <- data.frame(column = c(1, NA, 3), row.names = rownames(exprs_name_ok))
    obs_name_ok <- data.frame(column = c(1, 3), row.names = colnames(exprs_name_ok))
    ds_name_ok <- new("ExprDataSet", exprs = exprs_name_ok, obs = obs_name_ok, var = var_name_ok, uns = uns)

    exprs_idx_ok <- matrix(c(
        2, 0,
        0, 5
    ), ncol = 2, byrow = TRUE)
    rownames(exprs_idx_ok) <- c("a", "c")
    colnames(exprs_idx_ok) <- c("d", "f")
    var_idx_ok <- data.frame(column = c(1, 3), row.names = rownames(exprs_idx_ok))
    obs_idx_ok <- data.frame(column = c(1, 3), row.names = colnames(exprs_idx_ok))
    ds_idx_ok <- new("ExprDataSet", exprs = exprs_idx_ok, obs = obs_idx_ok, var = var_idx_ok, uns = uns)

    exprs_norm_ok <- matrix(c(
        40, 20, 0,
        60, 0, 0,
        0, 80, 100
    ), ncol = 3, byrow = TRUE)
    rownames(exprs_norm_ok) <- rownames(ds)
    colnames(exprs_norm_ok) <- colnames(ds)
    obs_norm_ok <- obs
    var_norm_ok <- var
    ds_norm_ok <- new("ExprDataSet", exprs = exprs_norm_ok, obs = obs_norm_ok, var = var_norm_ok, uns = uns)
})
ok_group("Dense ExprDataSet", {
    ok(all.equal(nrow(ds), 3), "nrow")
    ok(all.equal(ncol(ds), 3), "ncol")
    ok(all.equal(dim(ds), c(3, 3)), "dim")
    ok(all.equal(rownames(ds), rownames(ds@var)), "rownames")
    ok(all.equal(colnames(ds), rownames(ds@obs)), "colnames")
    ok(all.equal(ds, read_dataset("test.h5")), "Read and write dataset")
    ok(is(to_seurat(ds), "seurat"), "Convert to seurat")
    ok(all.equal(ds_name_ok, ds[c("a", "g", "c"), c("d", "f"), silent = TRUE]), "Name slicing")
    ok(all.equal(ds_idx_ok, ds[as.integer(c(1, 3)), as.integer(c(1, 3))]), "Integer slicing")
    ok(all.equal(ds_idx_ok, ds[c(TRUE, FALSE, TRUE), c(TRUE, FALSE, TRUE)]), "Bool slicing")
    ok(all.equal(ds_norm_ok, normalize(ds, target = 100)), "Normalize dataset")
})

suppressMessages({
    exprs <- as(exprs, "dgCMatrix")
    ds <- new("ExprDataSet", exprs = exprs, obs = obs, var = var, uns = uns)
    write_dataset(ds, "test.h5")

    exprs_name_ok <- as(exprs_name_ok, "dgCMatrix")
    ds_name_ok <- new("ExprDataSet", exprs = exprs_name_ok, obs = obs_name_ok, var = var_name_ok, uns = uns)

    exprs_idx_ok <- as(exprs_idx_ok, "dgCMatrix")
    ds_idx_ok <- new("ExprDataSet", exprs = exprs_idx_ok, obs = obs_idx_ok, var = var_idx_ok, uns = uns)

    exprs_norm_ok <- as(exprs_norm_ok, "dgCMatrix")
    ds_norm_ok <- new("ExprDataSet", exprs = exprs_norm_ok, obs = obs_norm_ok, var = var_norm_ok, uns = uns)
})
ok_group("Sparse ExprDataSet", {
    ok(all.equal(ds, read_dataset("test.h5")), "Read and write dataset")
    ok(all.equal(ds_name_ok, ds[c("a", "g", "c"), c("d", "f"), silent = TRUE]), "Name slicing")
    ok(all.equal(ds_idx_ok, ds[as.integer(c(1, 3)), as.integer(c(1, 3))]), "Integer slicing")
    ok(all.equal(ds_idx_ok, ds[c(TRUE, FALSE, TRUE), c(TRUE, FALSE, TRUE)]), "Bool slicing")
    ok(all.equal(ds_norm_ok, normalize(ds, target = 100)), "Normalize dataset")
})


file.remove("test.h5")
