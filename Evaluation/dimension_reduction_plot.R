#!/usr/bin/env Rscript

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(jsonlite)
    library(ggplot2)
    library(ggsci)
    library(reshape2)
    library(dplyr)
    library(extrafont)
})
source("../Utilities/utils.R")

# Read data
df <- read.csv(snakemake@input[["data"]], check.names = FALSE, stringsAsFactors = FALSE)
df$dataset <- factor(df$dataset, levels = df %>%
    select(dataset, n_cell) %>%
    arrange(n_cell) %>%
    distinct() %>%
    pull(dataset)
)  # This determines dataset order
df$method <- factor(df$method, levels = snakemake@config[["method"]])
levels(df$method) <- gsub("[_.]", " ", levels(df$method))
color_mapping <- unlist(fromJSON(snakemake@input[["palette"]]))

# MAP
gp <- ggplot(data = df %>% group_by(dataset, method, dimensionality) %>% summarise(
    sd = sd(mean_average_precision),
    mean_average_precision = mean(mean_average_precision)
), mapping = aes(
    x = dimensionality, y = mean_average_precision,
    ymin = mean_average_precision - sd,
    ymax = mean_average_precision + sd,
    col = method
)) + geom_line(alpha = 0.8) + geom_errorbar(width = 1.5, alpha = 0.6) + facet_wrap(
    ~dataset, nrow = 2
) + scale_color_manual(
    name = "Method", values = color_mapping, limits = levels(df$method)
) + scale_x_continuous(
    name = "Dimensionality"
) + scale_y_continuous(
    name = "Mean average precision"
)
ggsave(snakemake@output[["map"]], mod_style(gp), width = 7.5, height = 5)

# Optimal MAP
optdims <- as.data.frame(df %>% group_by(method, dimensionality, dataset) %>% summarise(
    mean_average_precision = mean(mean_average_precision)
) %>% group_by(method, dimensionality) %>% summarise(
    mean_average_precision = mean(mean_average_precision)
) %>% group_by(method) %>% summarise(
    dimensionality = dimensionality[which.max(mean_average_precision)]
))
optdims[optdims$method == "Cell BLAST", "dimensionality"] <- 10
method_dim <- sprintf("%s (%d)", optdims$method, optdims$dimensionality)
names(method_dim) <- optdims$method
optdf <- merge(optdims, df)
levels(optdf$method) <- method_dim[levels(optdf$method)]
names(color_mapping) <- method_dim[names(color_mapping)]

gp <- ggplot(data = optdf %>% group_by(dataset, method) %>% summarise(
    sd = sd(mean_average_precision),
    mean_average_precision = mean(mean_average_precision)
), mapping = aes(
    x = dataset, y = mean_average_precision,
    ymin = mean_average_precision - sd,
    ymax = mean_average_precision + sd,
    fill = method
)) + geom_bar(
    stat = "identity", position = position_dodge(0.85), width = 0.85
) + geom_errorbar(
    position = position_dodge(0.85), width = 0.2
) + scale_x_discrete(
    name = "Dataset"
) + scale_y_continuous(
    name = "Mean average precision"
) + scale_fill_manual(
    name = "Method", values = color_mapping
) + coord_cartesian(ylim = c(0.85, 1.0))
ggsave(snakemake@output[["optmap"]], mod_style(gp), width = 10, height = 4.5)

# Integrative
optdf_summarize_seed <- optdf %>% group_by(method, dataset) %>% summarise(
    mean_average_precision = mean(mean_average_precision)
) %>% arrange(method, dataset) %>% as.data.frame()
optdf_summarize_dataset <- optdf_summarize_seed %>% group_by(method) %>% summarise(
    stdev = sd(mean_average_precision),
    mean_average_precision = mean(mean_average_precision)
) %>% as.data.frame()
gp <- ggplot(data = optdf_summarize_dataset, mapping = aes(
    x = method, y = mean_average_precision, fill = method,
    ymin = mean_average_precision - stdev,
    ymax = mean_average_precision + stdev
)) + geom_bar(
    stat = "identity", width = 0.65
) + geom_errorbar(
    width = 0.15
) + scale_x_discrete(
    name = "Method", limits = optdf_summarize_dataset %>% arrange(
        desc(mean_average_precision)
    ) %>% pull(method)
) + scale_y_continuous(
    name = "Mean average precision"
) + scale_fill_manual(
    values = color_mapping, name = "Method"
) + coord_cartesian(ylim = c(0.85, 1.0)) + guides(fill = FALSE)
ggsave(snakemake@output[["integrative"]], mod_style(gp, rotate.x = TRUE), width = 5, height = 4)
