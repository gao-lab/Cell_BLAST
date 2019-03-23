#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(ggplot2)
    library(reshape2)
    library(dplyr)
    library(ggsci)
})
source("../Utilities/utils.R")

df <- read.csv("../Results/benchmark_bias_removal.csv",
               check.names = FALSE, stringsAsFactors = FALSE)
df$Dataset <- factor(df$Dataset, levels = df %>%
    select(Dataset, `Number of cells`) %>%
    arrange(`Number of cells`) %>%
    distinct() %>%
    pull(Dataset)
)  # This determines dataset order
levels(df$Dataset) <- gsub("\\+", "\n", levels(df$Dataset))
df$Regularization <- factor(
    df$Regularization, levels = sort(unique(df$Regularization)))
if (Sys.getenv("methods") != "") {
    df$Method <- factor(df$Method, levels = strsplit(
        Sys.getenv("methods"), " "
    )[[1]])
    levels(df$Method) <- gsub("_", " ", levels(df$Method))
}  # Order in the environment variable determines the color order

color_mapping <- pal_d3("category10")(nlevels(df$Regularization))
names(color_mapping) <- levels(df$Regularization)

fit <- list()
for (dataset in unique(df$Dataset)) {
    this_df <- df %>% filter(
        Dataset == dataset, Method == "Cell BLAST"
    ) %>% arrange(`Mean Average Precision`)
    if (nrow(this_df) == 0) next
    this_fit <- rotated.spline(
        this_df[, c("Mean Average Precision", "Seurat Alignment Score")],
        angle = - pi / 16, spar = 0.2, nknots = 5
    )
    this_fit$Dataset <- dataset
    fit[[length(fit) + 1]] <- this_fit
}
fit <- Reduce(rbind, fit)

gp <- ggplot(df %>% sample_frac(), mapping = aes(
    x = `Mean Average Precision`, y = `Seurat Alignment Score`
)) + geom_path(data = fit, size = 1.5, alpha = 0.15) + geom_point(
    mapping = aes(col = `Regularization`, shape = `Method`),
    size = 2.5, stroke = 0, alpha = 0.7
) + scale_shape_manual(values = 15:19) + scale_colour_manual(
    values = color_mapping, na.value = "#999999"
) + theme(
    strip.text.x = element_text(size = 7),
    legend.box = "horizontal"
) + facet_wrap(~Dataset, nrow = 1)
ggsave("../Results/benchmark_bias_removal_two_way.pdf",
       mod_style(gp), width = 12, height = 3)

color_mapping <- pal_d3("category10")(nlevels(df$Method))
names(color_mapping) <- levels(df$Method)

df <- df %>% filter(Regularization == 0.01 | Method != "Cell BLAST")
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
        Regularization = rep(NA, length(missing_methods)),
        Trial = rep(NA, length(missing_methods)),
        `Run time` = rep(NA, length(missing_methods)),
        `Mean Average Precision` = rep(-1, length(missing_methods)),
        `Seurat Alignment Score` = rep(-1, length(missing_methods)),
        `Batch Mixing Entropy` = rep(-1, length(missing_methods)),
        check.names = FALSE
    )
    df <- rbind(df, placeholder_df[, colnames(df)])
}  # leaving out blank spaces for missing categories

df$missing_label <- ""
df[is.na(df$Trial), "missing_label"] <- "N.A."

gp <- ggplot(df %>% group_by(Method, Dataset) %>% summarise(
    `Mean Average Precision SD` = sd(`Mean Average Precision`),
    `Mean Average Precision` = mean(`Mean Average Precision`),
    missing_label = unique(missing_label)
), aes(
    x = Dataset, y = `Mean Average Precision`, fill = Method,
    ymin = `Mean Average Precision` - `Mean Average Precision SD`,
    ymax = `Mean Average Precision` + `Mean Average Precision SD`
)) + geom_bar(
    stat = "identity", position = position_dodge(0.9), col = "black"
) + geom_errorbar(
    width = 0.3, position = position_dodge(0.9)
) + geom_text(
    aes(label = missing_label), fontface = "bold",
    y = 0.01, hjust = 0, angle = 90, size = 2.8, position = position_dodge(0.9)
) + scale_fill_manual(values = color_mapping) + scale_y_continuous(limits=c(0, NA))
ggsave("../Results/benchmark_bias_removal_mean_average_precision.pdf",
       mod_style(gp), width = 6.5, height = 5.5)

gp <- ggplot(df %>% group_by(Method, Dataset) %>% summarise(
    `Seurat Alignment Score SD` = sd(`Seurat Alignment Score`),
    `Seurat Alignment Score` = mean(`Seurat Alignment Score`),
    missing_label = unique(missing_label)
), aes(
    x = Dataset, y = `Seurat Alignment Score`, fill = Method,
    ymin = `Seurat Alignment Score` - `Seurat Alignment Score SD`,
    ymax = `Seurat Alignment Score` + `Seurat Alignment Score SD`
)) + geom_bar(
    stat = "identity", position = position_dodge(0.9), col = "black"
) + geom_text(
    aes(label = missing_label), fontface = "bold",
    y = 0.01, hjust = 0, angle = 90, size = 3, position = position_dodge(0.9)
) + geom_errorbar(
    width = 0.3, position = position_dodge(0.9)
) + scale_fill_manual(values = color_mapping) + scale_y_continuous(limits=c(0, NA))
ggsave("../Results/benchmark_bias_removal_seurat_alignment_score.pdf",
       mod_style(gp), width = 6.5, height = 5.5)

gp <- ggplot(df %>% group_by(Method, Dataset) %>% summarise(
    `Batch Mixing Entropy SD` = sd(`Batch Mixing Entropy`),
    `Batch Mixing Entropy` = mean(`Batch Mixing Entropy`),
    missing_label = unique(missing_label)
), aes(
    x = Dataset, y = `Batch Mixing Entropy`, fill = Method,
    ymin = `Batch Mixing Entropy` - `Batch Mixing Entropy SD`,
    ymax = `Batch Mixing Entropy` + `Batch Mixing Entropy SD`
)) + geom_bar(
    stat = "identity", position = position_dodge(0.9), col = "black"
) + geom_errorbar(
    width = 0.3, position = position_dodge(0.9)
) + geom_text(
    aes(label = missing_label), fontface = "bold",
    y = 0.01, hjust = 0, angle = 90, size = 3, position = position_dodge(0.9)
) + scale_fill_manual(values = color_mapping) + scale_y_continuous(limits=c(0, NA))
ggsave("../Results/benchmark_bias_removal_batch_mixing_entropy.pdf",
       mod_style(gp), width = 6.5, height = 5.5)

# gp <- ggplot(df %>% group_by(Method, Dataset) %>% summarise(
#     `Run time SD` = sd(`Run time`), `Run time (s)` = mean(`Run time`),
#     `Number of cells` = unique(`Number of cells`)
# ), aes(
#     x = `Number of cells`, y = `Run time (s)`, col = Method,
#     ymin = `Run time (s)` - `Run time SD`, ymax = `Run time (s)` + `Run time SD`
# )) + geom_point() + geom_line() + geom_errorbar(width = 0.03) + scale_colour_manual(values = color_mapping)
# ggsave("../Results/benchmark_bias_removal_run_time.pdf",
#        mod_style(gp, log.x = TRUE, log.y = TRUE), width = 5, height = 4)
