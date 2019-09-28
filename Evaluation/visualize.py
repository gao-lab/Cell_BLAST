import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Cell_BLAST as cb
import utils
plt.rcParams['svg.fonttype'] = "none"
plt.rcParams['font.family'] = "Arial"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", dest="x", type=str, required=True)
    parser.add_argument("-d", dest="data", type=str, required=True)
    parser.add_argument("-o", dest="output", type=str, required=True)

    parser.add_argument("-l", dest="label", type=str, required=True)
    parser.add_argument("-v", dest="vis", type=str, default="tSNE")
    parser.add_argument("-s", dest="shuffle", default=False, action="store_true")
    parser.add_argument("-r", dest="rasterized", default=False, action="store_true")
    parser.add_argument("-p", dest="psize", type=float, default=1.0)
    parser.add_argument("--width", dest="width", type=float, default=5.0)
    parser.add_argument("--height", dest="height", type=float, default=5.0)
    cmd_args = parser.parse_args()
    cmd_args.output = [cmd_args.output]
    cmd_args.input = argparse.Namespace(
        x=cmd_args.x,
        data=cmd_args.data
    )
    cmd_args.wildcards = argparse.Namespace(
        label=cmd_args.label,
        vis=cmd_args.vis
    )
    cmd_args.params = argparse.Namespace(
        shuffle=cmd_args.shuffle,
        psize=cmd_args.psize,
        width=cmd_args.width,
        height=cmd_args.height,
        rasterized=cmd_args.rasterized
    )
    del cmd_args.x, cmd_args.data, cmd_args.label, cmd_args.vis, \
        cmd_args.shuffle, cmd_args.psize, cmd_args.width, cmd_args.height, \
        cmd_args.rasterized
    return cmd_args


def main():
    x = cb.data.read_hybrid_path("//".join([
        snakemake.input.x, "visualization"
    ]))
    ds = cb.data.ExprDataSet.read_dataset(snakemake.input.data)
    ds = utils.clean_dataset(ds, snakemake.config["label"])

    axis1 = "{vis}1".format(vis=snakemake.wildcards.vis)
    axis2 = "{vis}2".format(vis=snakemake.wildcards.vis)
    label = snakemake.wildcards.label.replace("_", " ").capitalize()

    df = pd.DataFrame({
        axis1: x[:, 0],
        axis2: x[:, 1],
        label: pd.Categorical(
            ds.obs[snakemake.wildcards.label].values,
            categories=sorted(
                np.unique(ds.obs[snakemake.wildcards.label].values).tolist(),
                key=lambda x: x.lower()
            )
        )
    })
    if snakemake.params["shuffle"]:
        df = df.sample(frac=1)

    fig, ax = plt.subplots(figsize=(snakemake.params["width"], snakemake.params["height"]))
    ax = sns.scatterplot(
        x=axis1, y=axis2, hue=label, data=df,
        s=snakemake.params["psize"], edgecolor=None,
        rasterized=snakemake.params["rasterized"], ax=ax
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.legend(
        bbox_to_anchor=(1.05, 0.5), loc="center left",
        borderaxespad=0.0, frameon=False, prop=dict(
            size=snakemake.params["legend_size"]
        ), markerscale=snakemake.params["marker_scale"],
        labelspacing=snakemake.params["label_spacing"],
        ncol=np.ceil(np.unique(df[label]).size / 50).astype(np.int)
    )
    fig.savefig(snakemake.output[0], dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main()
