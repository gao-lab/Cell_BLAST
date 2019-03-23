#! /usr/bin/env Rscript
# by caozj
# 6 Nov 2017
# 4:08:25 PM

# This script make plots based on the learned latent representation

suppressPackageStartupMessages({
    library(argparse)
    library(rhdf5)
    library(dplyr)
    library(ggplot2)
    library(Rtsne)
})
source("data.R")


#===============================================================================
#
#  Parse arguments
#
#===============================================================================
parser <- ArgumentParser()

# Input
parser$add_argument("-i", "--input", dest = "input", type = "character", required = TRUE)
parser$add_argument("-m", "--method", dest = "method", default = "tSNE")
parser$add_argument("-l", "--label", dest = "lbl", type = "character", default = NULL)
parser$add_argument("-c", "--clean", dest = "clean", type = "character", default = NULL)
parser$add_argument("--shuffle", dest = "shuffle", default = FALSE, action = "store_true")

# Discrete
parser$add_argument("--ignore-label", dest = "ignore_lbl", type = "character", nargs = "+", default = NULL)
parser$add_argument("--merge-label", dest = "merge_lbl", type = "character", nargs = "+", default = NULL)
parser$add_argument("-n", "--hide-na", dest = "hide_na", default = FALSE, action = "store_true")
parser$add_argument("-s", "--separately", dest = "separately", default = FALSE, action = "store_true")
parser$add_argument("--order", dest = "order", default = FALSE, action = "store_true")

# Continuous
parser$add_argument("--force-continuous", dest = "force_continuous", action = "store_true", default = FALSE)
parser$add_argument("--mid", dest = "midpoint", type = "double", default = NULL)
parser$add_argument("-r", "--rev", dest = "reverse", action = "store_true", default = FALSE)

# Style
parser$add_argument("-p", "--psize", dest = "psize", type = "double", default = 1)
parser$add_argument("-a", "--alpha", dest = "alpha", type = "double", default = 1)
parser$add_argument("--no-legend", dest="no_legend", default = FALSE, action = "store_true")
parser$add_argument("--legend-pos", dest="legend_pos", type = "character", default = "right")
parser$add_argument("--legend-col", dest="legend_col", type = "integer", default = NULL)
parser$add_argument("--legend-text", dest="legend_text", type = "double", default = 9)
parser$add_argument("--keyheight", dest="keyheight", type = "double", default = NULL)
parser$add_argument("--height", dest = "height", type = "double", default = 6)
parser$add_argument("--width", dest = "width", type = "double", default = 6)

args <- parser$parse_args()


#===============================================================================
#
#  Read data
#
#===============================================================================
x <- t(read_hybrid_path(args$input))
x_split <- strsplit(args$input, "//")[[1]]
output_prefix <- file.path(
    dirname(x_split[1]),
    paste(gsub("\\.[^\\.]+$", "", basename(x_split[1])),
          basename(x_split[2]), sep = "_")
)

if (!is.null(args$lbl)) {
    if (grepl("//var/", args$lbl)) {  # Feature plot
        lbl_split <- strsplit(args$lbl, "//")[[1]]
        dataset <- read_dataset(lbl_split[1])
        dataset <- normalize(dataset)
        lbl <- log1p(as.vector(dataset[basename(lbl_split[2]), ]@exprs))
    } else {
        lbl <- as.vector(read_hybrid_path(args$lbl))
    }
    if (!is.null(args$clean)) {
        clean <- as.vector(read_hybrid_path(args$clean))
        lbl <- lbl[!(
            clean %in% c("", "NA", "na", "NaN", "nan") |
            is.na(clean) | is.nan(clean)
        )]
    }
    stopifnot(length(lbl) == nrow(x))
    discrete_flag <- is.factor(lbl) || is.character(lbl) || all((lbl %% 1) == 0)
    discrete_flag <- discrete_flag && ! args$force_continuous
    if (discrete_flag) {
        for (item in args$merge_lbl) {
            if (grepl("^@", item)) {
                replacement <- sub("@", "", item)
            } else {
                lbl[grepl(item, lbl)] <- replacement
            }
        }
        if (!is.null(args$ignore_lbl)) {
            mask <- Reduce(
                `|`, lapply(
                    as.list(args$ignore_lbl),
                    function(pattern) grepl(pattern, lbl)
                )
            )
            lbl[mask] <- NA
        }
    }
    name <- basename(args$lbl)
}


#===============================================================================
#
#  Dimension reduction
#
#===============================================================================
if (ncol(x) > 2) {
    cached_hybrid_path <- sprintf(
        "%s//%s/%s", x_split[1], args$method, x_split[2])
    if (!check_hybrid_path(cached_hybrid_path))
        system(sprintf("./visualize.py -i %s -m %s", args$input, args$method))
    x <- t(read_hybrid_path(cached_hybrid_path))
    axis_title <- args$method
} else {
    axis_title <- "DIM"
}


#===============================================================================
#
#  Plotting
#
#===============================================================================
gp_beautify <- function(gp) {
    gp + scale_x_continuous(name = paste0(axis_title, "1")) +
    scale_y_continuous(name = paste0(axis_title, "2")) +
    theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        legend.position = args$legend_pos,
        legend.text = element_text(size = args$legend_text),
        legend.title = element_text(size = args$legend_text + 1)
        # plot.margin = unit(c(0.1, 0.8, 0.1, 0.2), "in")
    )
}

if (is.null(args$lbl)) {
    gp <- ggplot(data.frame(
        DIM1 = x[, 1], DIM2 = x[, 2]
    ), aes(x = DIM1, y = DIM2))
    gp1 <- gp + geom_point(size = args$psize, alpha = args$alpha)
    gp2 <- gp + stat_density_2d(
        geom = "raster", aes(fill = ..density..),
        contour = FALSE, interpolate = TRUE
    ) + scale_fill_gradient2(
        low = "#003263", high = "#780222", mid = "#DDDDDD",
        midpoint = 0.3, name = "density"
    )
    ggsave(sprintf("%s_%s.pdf", output_prefix, args$method), gp_beautify(gp1),
           height = args$height, width = args$width)
    ggsave(sprintf("%s_%s_density.pdf", output_prefix, args$method), gp_beautify(gp2),
           height = args$height, width = args$width)
} else {
    df <- data.frame(DIM1 = x[, 1], DIM2 = x[, 2], lbl = lbl)
    if (args$hide_na)
        df <- df %>% filter(!is.na(lbl))
    if (args$shuffle)
        df <- df[sample(1:nrow(df), nrow(df), replace = FALSE), ]
    if (!discrete_flag) {
        midpoint <- if (is.null(args$midpoint)) median(lbl) else args$midpoint
        if (!args$shuffle)
            df <- df[order(df$lbl, decreasing = args$rev), ]
        gp <- ggplot(df, aes(
            x = DIM1, y = DIM2, col = lbl
        )) + geom_point(
            size = args$psize, alpha = args$alpha
        ) + scale_color_gradient2(
            low = "#003263", high = "#780222", mid = "#DDDDDD",
            midpoint = midpoint, name = name
        )
        ggsave(sprintf("%s_%s_%s.pdf", output_prefix, args$method, name),
               gp_beautify(gp), height = args$height, width = args$width)
    } else {
        # Separately
        if (args$separately) {
            for (single_lbl in unique(df$lbl)) {
                ordering <- order(lbl == single_lbl)
                gp <- ggplot(df[ordering, ], aes(
                    x = DIM1, y = DIM2,
                    col = lbl == single_lbl
                )) + scale_color_manual(
                    values = c("#989898", "#C56637")
                ) + geom_point(
                    size = args$psize, alpha = args$alpha
                ) + guides(col = FALSE)
                ggsave(sprintf("%s_%s_%s_%s.jpg",
                    output_prefix, args$method, name,
                    gsub("[/\ ]", "_", single_lbl)
                ), gp_beautify(gp), height = args$height, width = args$width)
            }
        } else {
            # Altogether
            if (args$order)
                df <- df[order(df$lbl, decreasing = args$rev), ]
            df$lbl <- as.factor(df$lbl)
            gp <- ggplot(df, aes(
                x = DIM1, y = DIM2, col = lbl
            )) + geom_point(
                size = args$psize, alpha = args$alpha
            ) + scale_color_discrete(name = name) + guides(
                col = guide_legend(
                    ncol = args$legend_col,
                    keyheight = args$keyheight
                )
            )
            if (args$no_legend)
                gp <- gp + guides(col = FALSE)
            ggsave(sprintf("%s_%s_%s.pdf", output_prefix, args$method, name),
                   gp_beautify(gp), height = args$height, width = args$width)
        }
    }
}

message("Done!")
