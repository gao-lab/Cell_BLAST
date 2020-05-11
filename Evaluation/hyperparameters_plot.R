#!/usr/bin/env Rscript

source("../.Rprofile", chdir = TRUE)
suppressPackageStartupMessages({
    library(ggplot2)
    library(ggsci)
    library(reshape2)
    library(dplyr)
    library(ggsci)
    library(ggpubr)
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

facets <- c(
    "dimensionality", "hidden_layer", "depth",
    "cluster", "lambda_prior", "prob_module"
)
df_list <- list()
for (facet in facets) {
    mask <- Reduce(`&`, lapply(
        setdiff(facets, facet),
        function(item) df[[item]] == snakemake@config[[item]][["default"]]
    ))
    df_list[[facet]] <- df[mask, c(facet, setdiff(colnames(df), facets))]
    df_list[[facet]] <- melt(df_list[[facet]], measure.vars = facet)
}
df <- Reduce(rbind, df_list)
df_val_levels <- unique(df$value)
df_val_rank <- integer(length(df_val_levels))
suppressWarnings(mask <- !is.na(as.numeric(df_val_levels)))
df_val_rank[mask] <- order(as.numeric(df_val_levels[mask]))
df_val_rank[!mask] <- order(df_val_levels[!mask]) + sum(mask)
df$value <- factor(df$value, levels = df_val_levels[df_val_rank])

color_mapping <- pal_d3("category10")(nlevels(df$dataset))

gp <- ggplot(data = df %>% group_by(dataset, variable, value) %>% summarise(
    sd = sd(mean_average_precision),
    mean_average_precision = mean(mean_average_precision)
), mapping = aes(
    x = value, y = mean_average_precision,
    ymin = mean_average_precision - sd,
    ymax = mean_average_precision + sd,
    group = dataset, col = dataset
)) + geom_line() + geom_errorbar(
    width = 0.1
) + facet_wrap(
    ~variable, scales = "free_x"
) + scale_x_discrete(
    name = "Hyperparameter value"
) + scale_y_continuous(
    name = "Mean average precision"
) + scale_color_manual(
    name = "Dataset", values = color_mapping
)
ggsave(snakemake@output[["map"]], mod_style(gp), width = 7.5, height = 5)

df$dataset <- factor(df$dataset, levels = snakemake@config[["dataset"]])
dataset_meta <- rbind(
    read.csv(
        "../Datasets/ACA_datasets.csv", row.names = 1, comment = "#",
        check.names = FALSE, stringsAsFactors = FALSE
    )[, "platform", drop = FALSE],
    read.csv(
        "../Datasets/additional_datasets.csv", row.names = 1, comment = "#",
        check.names = FALSE, stringsAsFactors = FALSE
    )[, "platform", drop = FALSE]
)
levels(df$dataset) <- sapply(levels(df$dataset), function(x) {
    sprintf("%s\n(%s)", x, dataset_meta[x, "platform"])
})
prob_df <- df %>% filter(variable == "prob_module")
prob_df_blank <- prob_df %>% mutate(
    mean_average_precision = mean_average_precision + 0.0002
)  # Slighly increase the gap between significance labels and boxes
gp <- ggplot(data = prob_df, mapping = aes(
    x = value, y = mean_average_precision,
    col = value, fill = value
)) + geom_boxplot(alpha = 0.5, width = 0.5) + facet_wrap(
    ~dataset, scales = "free_y", ncol = 3
) + stat_compare_means(
    method = "wilcox.test", label = "p.signif", label.x.npc = "center", size = 3.5
) + geom_blank(data = prob_df_blank) + scale_x_discrete(
    name = "Generative distribution"
) + scale_y_continuous(
    name = "Mean average precision"
) + scale_fill_d3() + scale_color_d3() + guides(fill = FALSE, color = FALSE)
ggsave(snakemake@output[["probmap"]], mod_style(gp), width = 7, height = 8)
