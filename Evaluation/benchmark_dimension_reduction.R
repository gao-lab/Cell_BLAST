#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(ggplot2)
    library(ggsci)
    library(reshape2)
    library(dplyr)
    library(ggsci)
})
source("../Utilities/utils.R")

df <- read.csv("../Results/benchmark_dimension_reduction.csv",
               check.names = FALSE, stringsAsFactors = FALSE)

df$Dataset <- factor(df$Dataset, levels = df %>%
    select(Dataset, `Number of cells`) %>%
    arrange(`Number of cells`) %>%
    distinct() %>%
    pull(Dataset)
)  # This determines dataset order
if (Sys.getenv("methods") != "") {
    df$Method <- factor(df$Method, levels = rev(strsplit(
        Sys.getenv("methods"), " "
    )[[1]]))
    levels(df$Method) <- gsub("_", " ", levels(df$Method))
}  # Order in the environment variable determines the color order

for (dataset in unique(df$Dataset)) {
    missing_methods <- setdiff(
        levels(df$Method),
        unique(df %>% filter(Dataset == dataset) %>% pull(Method))
    )
    placeholder_df <- data.frame(
        Method = missing_methods,
        Dataset = rep(dataset, length(missing_methods)),
        `Number of cells` = rep(NA, length(missing_methods)),
        Genes = rep(NA, length(missing_methods)),
        Conf = rep(NA, length(missing_methods)),
        Trial = rep(NA, length(missing_methods)),
        `Run time` = rep(NA, length(missing_methods)),
        `Nearest Neighbor Accuracy` = rep(-1, length(missing_methods)),
        `Mean Average Precision` = rep(-1, length(missing_methods)),
        check.names = FALSE
    )
    df <- rbind(df, placeholder_df[, colnames(df)])
}  # leaving out blank spaces for missing categories
df$missing_label <- ""
df[is.na(df$Trial), "missing_label"] <- "N.A."

bg_rects <- data.frame(
    xmin = seq(nlevels(df$Dataset)) - 0.5,
    xmax = seq(nlevels(df$Dataset)) + 0.5,
    bgc = rep_len(c("dark", "light"), nlevels(df$Dataset))
)
color_mapping <- pal_d3("category10")(nlevels(df$Method))
names(color_mapping) <- rev(levels(df$Method))
color_mapping["light"] <- "#FFFFFF"
color_mapping["dark"] <- "#EEEEEE"

gp <- ggplot(data = df %>% group_by(Dataset, Method) %>% summarise(
    sd = sd(`Nearest Neighbor Accuracy`),
    `Nearest Neighbor Accuracy` = mean(`Nearest Neighbor Accuracy`),
    missing_label = unique(missing_label)
), mapping = aes(
    x = Dataset, y = `Nearest Neighbor Accuracy`,
    ymin = `Nearest Neighbor Accuracy` - sd,
    ymax = `Nearest Neighbor Accuracy` + sd,
    fill = Method
)) + scale_x_discrete(limits = levels(df$Dataset)) +
geom_rect(data = bg_rects, mapping = aes(
    ymin = 0.0, ymax = 2.0, xmin = xmin, xmax = xmax, fill = bgc
), inherit.aes = FALSE) +
geom_bar(stat = "identity", position = position_dodge(0.75), width = 0.75) +
geom_errorbar(position = position_dodge(0.75), width = 0.2) +
geom_text(aes(label = missing_label), position = position_dodge(0.75), y = 0.9, size = 2.6, fontface = "bold") +
scale_fill_manual(
    values = color_mapping, limits = rev(levels(df$Method))
) + coord_flip(ylim = c(0.9, 1.0))
ggsave("../Results/benchmark_dimension_reduction_nearest_neighbor_accuracy.pdf",
       mod_style(gp), width = 7, height = 6)

gp <- ggplot(data = df %>% group_by(Dataset, Method) %>% summarise(
    sd = sd(`Mean Average Precision`),
    `Mean Average Precision` = mean(`Mean Average Precision`),
    missing_label = unique(missing_label)
), mapping = aes(
    x = Dataset, y = `Mean Average Precision`,
    ymin = `Mean Average Precision` - sd,
    ymax = `Mean Average Precision` + sd,
    fill = Method
)) + scale_x_discrete(limits = levels(df$Dataset)) +
geom_rect(data = bg_rects, mapping = aes(
    ymin = 0.0, ymax = 2.0, xmin = xmin, xmax = xmax, fill = bgc
), inherit.aes = FALSE) +
geom_bar(stat = "identity", position = position_dodge(0.75), width = 0.75) +
geom_errorbar(position = position_dodge(0.75), width = 0.2) +
geom_text(aes(label = missing_label), position = position_dodge(0.75), y = 0.9, size = 2.6, fontface = "bold") +
scale_fill_manual(
    values = color_mapping, limits = rev(levels(df$Method))
) + coord_flip(ylim = c(0.9, 1.0))
ggsave("../Results/benchmark_dimension_reduction_mean_average_precision.pdf",
       mod_style(gp), width = 7, height = 6)
