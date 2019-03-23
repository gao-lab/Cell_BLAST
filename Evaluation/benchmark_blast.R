#!/usr/bin/env Rscript

suppressPackageStartupMessages({
    library(ggplot2)
    library(reshape2)
    library(dplyr)
    library(ggsci)
    library(MESS)
})
source("../Utilities/utils.R")


#===============================================================================
#
#  Read data
#
#===============================================================================
df <- read.csv("../Results/benchmark_blast.csv",
               check.names = FALSE, stringsAsFactors = FALSE)
df$Organ <- factor(df$Organ, levels = df %>%
    select(Organ, `Reference size`) %>%
    arrange(`Reference size`) %>%
    distinct() %>%
    pull(Organ)
)  # Determine order by reference size
if (Sys.getenv("methods") != "") {
    df$Method <- factor(df$Method, levels = rev(strsplit(
        Sys.getenv("methods"), " "
    )[[1]]))
    levels(df$Method) <- gsub("_", " ", levels(df$Method))
}  # Order in the environment variable determines the color order

scmap_default_threshold <- as.numeric(Sys.getenv("scmap_default_threshold"))
cellfishing_default_threshold <- as.numeric(Sys.getenv("CellFishing_default_threshold"))
cell_blast_default_threshold <- as.numeric(Sys.getenv("Cell_BLAST_default_threshold"))
df$`Default Cutoff` <- FALSE
df$`Default Cutoff`[
    df$Method == "scmap" &
    df$Threshold == scmap_default_threshold
] <- TRUE
df$`Default Cutoff`[
    df$Method == "CellFishing.jl" &
    df$Threshold == cellfishing_default_threshold
] <- TRUE
df$`Default Cutoff`[
    df$Method %in% c("Cell BLAST", "Cell BLAST aligned") &
    df$Threshold == cell_blast_default_threshold
] <- TRUE

color_mapping <- pal_d3("category10")(nlevels(df$Method))
names(color_mapping) <- rev(levels(df$Method))


df_use_list <- list(
    include_align = df %>% filter(
        Dataset %in% strsplit(Sys.getenv("include_align"), " ")),
    no_align = df %>% filter(
        !Dataset %in% strsplit(Sys.getenv("include_align"), " "),
        !Method %in% "Cell BLAST aligned")
)
for (df_use_name in names(df_use_list)) {
    df_use <- droplevels(df_use_list[[df_use_name]])
    df_use_summary <- df_use %>% group_by(Method, Organ, Threshold) %>% summarise(
        Sensitivity = mean(Sensitivity),
        Specificity = mean(Specificity),
        Kappa = mean(Kappa)
    ) %>% mutate(SensSpecSum = Sensitivity + Specificity) %>% filter(
        SensSpecSum == max(SensSpecSum)
    ) %>% as.data.frame()
    fit <- list()
    for (method in unique(df_use$Method)) {
        for (organ in unique(df_use$Organ)) {
            this_df_use <- df_use %>% filter(
                Method == method, Organ == organ
            ) %>% arrange(Specificity)
            if (nrow(this_df_use) == 0) next

            df_use_summary[
                df_use_summary$Method == method & df_use_summary$Organ == organ, "AUC"
            ] <- auc(
                x = c(0, this_df_use$Specificity, 1),
                y = c(1, this_df_use$Sensitivity, 0),
                type = "linear"
            )

            this_fit <- rotated.spline(
                this_df_use[, c("Specificity", "Sensitivity")],
                angle = - pi / 5, spar = 0.3, nknots = 7
            )
            this_fit$Method <- method
            this_fit$Organ <- organ
            fit[[length(fit) + 1]] <- this_fit
        }
    }
    fit <- Reduce(rbind, fit)
    fit$Method <- factor(fit$Method, levels = levels(df_use$Method))

    wrapped_plot <- length(unique(df_use$Organ)) > 1
    inframe_text_size <- ifelse(wrapped_plot, 3, 4)
    inframe_line_height <- ifelse(wrapped_plot, 0.1, 0.08)
    gp <- ggplot() + geom_path(
        data = fit, mapping = aes(
            x = Specificity, y = Sensitivity, col = Method
        ), size = 1.5, alpha = 0.4
    ) + geom_point(
        data = df_use %>% sample_frac() %>% arrange(`Default Cutoff`),
        mapping = aes(
            x = Specificity, y = Sensitivity, col = Method,
            shape = `Default Cutoff`
        ), alpha = 0.7, stroke = 0, size = 3
    ) + geom_text(
        data = df_use_summary %>% mutate(
            label = sprintf("AUC: %.3f, Kappa: %.3f", AUC, Kappa),
            y = (length(unique((Method))) - as.integer(Method) + 1) * inframe_line_height
        ), mapping = aes(
            label = label, y = y, col = Method
        ), x = 0.1, size = inframe_text_size, hjust=0, show.legend = FALSE
    ) + scale_size_continuous(range = c(1, 4)) + scale_colour_manual(
        values = color_mapping, limits = rev(levels(df_use$Method))
    ) + scale_shape_discrete(name = "Default Cutoff")
    if (wrapped_plot) {
        gp <- gp + facet_wrap(~Organ, nrow = 2)  # + theme(legend.box = "horizontal")
        ggsave(sprintf("../Results/benchmark_blast_%s_roc.pdf", df_use_name), mod_style(gp), width = 8, height = 7)
    } else {
        ggsave(sprintf("../Results/benchmark_blast_%s_roc.pdf", df_use_name), mod_style(gp), width = 6, height = 4)
    }


    df_use <- rbind(
        df_use %>% filter(Method == "scmap", Threshold == scmap_default_threshold),
        df_use %>% filter(Method == "CellFishing.jl", Threshold == cellfishing_default_threshold),
        df_use %>% filter(Method %in% c("Cell BLAST", "Cell BLAST aligned"),
                          Threshold == cell_blast_default_threshold)
    )

    gp <- ggplot(df_use %>% group_by(Method, Organ) %>% summarise(
        SensitivitySD = sd(Sensitivity),
        Sensitivity = mean(Sensitivity)
    ), aes(
        x = Organ, y = Sensitivity, fill = Method,
        ymin = Sensitivity - SensitivitySD,
        ymax = Sensitivity + SensitivitySD
    )) + geom_bar(
        stat = "identity", position = position_dodge(0.7),
        col = "black", width = 0.7
    ) + geom_errorbar(
        width = 0.2, position = position_dodge(0.7)
    ) + coord_flip() + scale_fill_manual(
        values = color_mapping, limits = rev(levels(df_use$Method))
    )
    ggsave(sprintf("../Results/benchmark_blast_%s_sensitivity.pdf", df_use_name), mod_style(gp) + theme(
        axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))
    ), width = 7, height = 4)

    gp <- ggplot(df_use %>% group_by(Method, Organ) %>% summarise(
        SpecificitySD = sd(Specificity),
        Specificity = mean(Specificity)
    ), aes(
        x = Organ, y = Specificity, fill = Method,
        ymin = Specificity - SpecificitySD,
        ymax = Specificity + SpecificitySD
    )) + geom_bar(
        stat = "identity", position = position_dodge(0.7),
        col = "black", width = 0.7
    ) + geom_errorbar(
        width = 0.2, position = position_dodge(0.7)
    ) + coord_flip() + scale_fill_manual(
        values = color_mapping, limits = rev(levels(df_use$Method))
    )
    ggsave(sprintf("../Results/benchmark_blast_%s_specificity.pdf", df_use_name), mod_style(gp) + theme(
        axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))
    ), width = 7, height = 4)

    gp <- ggplot(df_use %>% group_by(Method, Organ) %>% summarise(
        KappaSD = sd(Kappa),
        Kappa = mean(Kappa)
    ), aes(
        x = Organ, y = Kappa, fill = Method,
        ymin = Kappa - KappaSD,
        ymax = Kappa + KappaSD
    )) + geom_bar(
        stat = "identity", position = position_dodge(0.7),
        col = "black", width = 0.7
    ) + geom_errorbar(
        width = 0.2, position = position_dodge(0.7)
    ) + coord_flip() + scale_fill_manual(
        values = color_mapping, limits = rev(levels(df_use$Method))
    )
    ggsave(sprintf("../Results/benchmark_blast_%s_kappa.pdf", df_use_name), mod_style(gp) + theme(
        axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))
    ), width = 7, height = 4)

    gp <- ggplot(df_use %>% group_by(Method, Organ) %>% summarise(
        `Extended Kappa SD` = sd(`Extended Kappa`),
        `Extended Kappa` = mean(`Extended Kappa`)
    ), aes(
        x = Organ, y = `Extended Kappa`, fill = Method,
        ymin = `Extended Kappa` - `Extended Kappa SD`,
        ymax = `Extended Kappa` + `Extended Kappa SD`
    )) + geom_bar(
        stat = "identity", position = position_dodge(0.7),
        col = "black", width = 0.7
    ) + geom_errorbar(
        width = 0.2, position = position_dodge(0.7)
    ) + coord_flip() + scale_fill_manual(
        values = color_mapping, limits = rev(levels(df_use$Method))
    )
    ggsave(sprintf("../Results/benchmark_blast_%s_extended_kappa.pdf", df_use_name), mod_style(gp) + theme(
        axis.title.y = element_text(margin = margin(t = 0, r = 20, b = 0, l = 0))
    ), width = 7, height = 4)

    # gp <- ggplot(df_use %>% group_by(Method, Organ) %>% summarise(
    #     `Time per query SD` = sd(`Time per query`),
    #     `Time per query (ms)` = mean(`Time per query`),
    #     `Reference size` = unique(`Reference size`)
    # ), aes(
    #     x = `Reference size`, y = `Time per query (ms)`, col = Method,
    #     ymin = `Time per query (ms)` - `Time per query SD`,
    #     ymax = `Time per query (ms)` + `Time per query SD`
    # )) + geom_point() + geom_line(size=1) + geom_errorbar(width = 120, size=0.6) + scale_fill_manual(
    #     values = color_mapping, limits = rev(levels(df_use$Method))
    # )
    # ggsave("../Results/benchmark_blast_time.pdf",
    #     mod_style(gp), width = 5, height = 4)
}
