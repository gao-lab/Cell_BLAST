suppressPackageStartupMessages({
    library(scales)
    library(ggplot2)
    library(extrafont)
})


# Shared style
mod_style <- function(
    gp, rotate.x = FALSE, log.x = FALSE, log.y = FALSE,
    axis.text.x.size = NULL, font.family = "Arial", ...
) {
    if (log.x)
        gp <- gp + scale_x_log10(
            breaks = trans_breaks("log10", function(x) 10 ^ x),
            labels = trans_format("log10", math_format(10 ^ .x))
        )
    if (log.y)
        gp <- gp + scale_y_log10(
            breaks = trans_breaks("log10", function(x) 10 ^ x),
            labels = trans_format("log10", math_format(10 ^ .x))
        )
    args <- list(
        text = element_text(family = font.family),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text.x = element_text(size=axis.text.x.size),
        ...
    )
    if (rotate.x)
        args$axis.text.x <- element_text(angle = 20, vjust = 0.6)
    gp <- gp + do.call(theme, args)
    gp
}


rotate.df <- function(x, angle) {
    stopifnot(ncol(x) == 2)
    rot <- matrix(c(
        cos(angle), - sin(angle),
        sin(angle), cos(angle)
    ), byrow = TRUE, nrow = 2, ncol = 2)
    rot.x <- as.matrix(x) %*% rot
    colnames(rot.x) <- colnames(x)
    as.data.frame(rot.x)
}


rotated.spline <- function(
    data, angle, n = 1000, ...
) {  # Assuming first column as x and second column as y
    data.rot <- rotate.df(data, angle)
    spline <- smooth.spline(x = data.rot[, 1], y = data.rot[, 2],  ...)
    pred.rot.x <- seq(
        from = min(data.rot[, 1]),
        to = max(data.rot[, 1]),
        length.out = n
    )
    pred.rot.y <- predict(spline, pred.rot.x)$y
    pred.rot <- cbind(pred.rot.x, pred.rot.y)
    pred.rot <- pred.rot[order(pred.rot[, 1]), ]
    colnames(pred.rot) <- colnames(data)
    pred.rot <- as.data.frame(pred.rot)
    rotate.df(pred.rot, -angle)
}


roc.smooth <- function(
    data, n = 1000, k = 5, L = 1.05, angle = - pi / 4, plot = FALSE, ...
) {
    sigmoid <- function (x) L * exp(k * x) / (exp(k * x) + 1)
    logit <- function (x) log((x / L) / (1 - x / L)) / k

    # Transform
    logit.data <- data.frame(x = logit(data[, 1]), y = logit(data[, 2]))
    logit.data[logit.data > 1] <- 1
    logit.data[logit.data < -1] <- -1
    # logit.data <- logit.data[apply(logit.data, 1, function(x) all(is.finite(x))), ]
    rot.logit.data <- rotate.df(logit.data, angle)

    # Fit
    fit <- smooth.spline(x = rot.logit.data$x, y = rot.logit.data$y, ...)
    x <- seq(min(rot.logit.data$x), max(rot.logit.data$x), length.out = n)
    y <- predict(fit, x)$y

    # require(mgcv)
    # fit <- gam(y ~ s(x, k = 10), data = rot.logit.data)
    # x <- seq(min(rot.logit.data$x), max(rot.logit.data$x), length.out = n)
    # y <- predict(fit, newdata = data.frame(x = x))

    rot.logit.pred <- data.frame(x = x, y = y)

    # Plot
    if (plot) {
        gp <- ggplot(data = NULL, mapping = aes(x = x, y = y)) +
            geom_point(data = rot.logit.data) +
            geom_line(data = rot.logit.pred, color = "grey", alpha = 0.5, size = 3)
        tmp <- basename(tempfile(fileext = ".pdf"))
        print(sprintf("Creating a temporary plot: %s", tmp))
        ggsave(tmp, gp)
    }

    # Inverse-transform
    logit.pred <- rotate.df(rot.logit.pred, -angle)
    pred <- data.frame(x = sigmoid(logit.pred$x), y = sigmoid(logit.pred$y))
    colnames(pred) <- colnames(data)
    pred
}
