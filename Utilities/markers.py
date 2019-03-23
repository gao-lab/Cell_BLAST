from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def fast_markers(mat, group, fnames,
                 alternative="two-sided", multitest="bonferroni",
                 min_pct=0.1, min_pct_diff=-np.inf,
                 logfc_threshold=0.25, pseudocount=1,
                 n_jobs=1):

    """
    Find markers for each group by one-vs-rest Wilcoxon rank sum test

    Parameters
    ----------
    mat : array_like
        n * m matrix, where n is obs number, m is feature number.
    group : array_like
        A sequence of labels used as obs grouping, length equal to n.
    fnames : array_like
        A sequence of feature names, length equal to m.
    alternative : str
        Alternative assumption, one of ["two-sided", "greater", "less"],
        default is "two-sided".
    multitest : str
        Method of multiple test p-value correction, default is "bonferroni".
    min_pct : float
        Minimal percent of cells expressing gene of interest, either in group or
        rest, for it to be considered in statistical test.
    min_pct_diff : float
        Minimal percent difference of cells expressing gene of interest in group
        and rest, for it to be considered in statistical test.
    logfc_threshold : float
        Minimal log fold change in average expression level of gene of interest,
        between group and rest, for it to be considered in statistical test.
    pseudocount : float
        Pseudocount to be added when computing log fold change.
    n_jobs : int
        Number of parallel running threads to use.

    Returns
    -------
    summary : dict
        Each element, named by group, is a pandas DataFrame containing
        differential expression results.
        Columns of each DataFrame are:
            "pct_1": percent of cells expressing the gene in group of interest
            "pct_2": percent of cells expressing the gene in the rest cells
            "logfc": log fold change of mean expression between group and rest
            "stat": statistic of Wilcoxon rank sum test
            "z": normal approximation statistic of Wilcoxon rank sum test
            "pval": p-value of Wilcoxon rank sum test
            "padj": p-value adjusted for multiple test
    """

    assert mat.shape[0] == len(group)
    label_encoder, onehot_encoder = LabelEncoder(), OneHotEncoder()
    group_onehot = onehot_encoder.fit_transform(
        label_encoder.fit_transform(group).reshape((-1, 1))
    ).astype(bool).toarray()
    labels = label_encoder.inverse_transform(np.arange(group_onehot.shape[1]))
    n_x = np.sum(group_onehot, axis=0).astype(np.float64)
    n_y = group_onehot.shape[0] - n_x
    n_xy_prod = n_x * n_y
    n_xy_plus = n_x + n_y

    def ranksum_thread(vec):

        """
        Wilcoxon rank sum test for one feature
        Adapted from the following R functions:
            `Seurat::FindMarkers` and `stats::wilcox.test`
        """

        # Preparation
        vec = vec.toarray().ravel() if issparse(vec) else vec
        pct = np.empty((group_onehot.shape[1], 2))
        logfc = np.empty((group_onehot.shape[1]))

        for i in range(group_onehot.shape[1]):
            mask = group_onehot[:, i].ravel()
            pct[i, 0] = round(np.sum(vec[mask] > 0) / n_x[i], 3)
            pct[i, 1] = round(np.sum(vec[~mask] > 0) / n_y[i], 3)
            logfc[i] = np.log(vec[mask].mean() + pseudocount) - \
                np.log(vec[~mask].mean() + pseudocount)

        # Percent expressed filtering
        pct_max = pct.max(axis=1)
        pct_min = pct.min(axis=1)
        pct_diff = pct_max - pct_min
        pct_mask = (pct_max > min_pct) & (pct_diff > min_pct_diff)

        # Fold change filtering
        if alternative == "greater":
            logfc_mask = logfc > logfc_threshold
        elif alternative == "less":
            logfc_mask = logfc < -logfc_threshold
        elif alternative == "two-sided":
            logfc_mask = abs(logfc) > logfc_threshold

        total_mask = pct_mask & logfc_mask
        if not np.any(total_mask):
            nan_placeholder = np.empty(group_onehot.shape[1])
            nan_placeholder[:] = np.nan
            return pct[:, 0].ravel(), pct[:, 1].ravel(), logfc, \
                nan_placeholder, nan_placeholder, nan_placeholder

        # Rank sum test
        rank = rankdata(vec)
        n_ties = np.unique(rank, return_counts=True)[1]

        stat = np.empty(group_onehot.shape[1])
        for i in range(group_onehot.shape[1]):
            mask = group_onehot[:, i].ravel()
            if total_mask[i]:
                stat[i] = rank[mask].sum() - n_x[i] * (n_x[i] + 1) / 2
            else:
                stat[i] = np.nan
        z = stat - n_x * n_y / 2
        sigma = np.sqrt((n_xy_prod / 12) * (
            (n_xy_plus + 1) -
            (n_ties ** 3 - n_ties).sum() / (n_xy_plus * (n_xy_plus - 1))
        ))
        if alternative == "two-sided":
            correction = np.sign(z) * 0.5
        elif alternative == "greater":
            correction = 0.5
        elif alternative == "less":
            correction = -0.5
        z = (z - correction) / sigma
        if alternative == "two-sided":
            pval = 2 * np.stack([norm.sf(z), norm.cdf(z)], axis=0).min(axis=0)
        elif alternative == "greater":
            pval = norm.sf(z)
        elif alternative == "less":
            pval = norm.cdf(z)
        return pct[:, 0].ravel(), pct[:, 1].ravel(), logfc, stat, z, pval

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        result = executor.map(ranksum_thread, mat.T)

    pct_1_list, pct_2_list, logfc_list, stat_list, z_list, pval_list = \
        [], [], [], [], [], []
    for pct_1, pct_2, logfc, stat, z, pval in result:
        pct_1_list.append(pct_1)
        pct_2_list.append(pct_2)
        logfc_list.append(logfc)
        stat_list.append(stat)
        z_list.append(z)
        pval_list.append(pval)
    pct_1 = np.stack(pct_1_list, axis=0).T
    pct_2 = np.stack(pct_2_list, axis=0).T
    logfc = np.stack(logfc_list, axis=0).T
    stat = np.stack(stat_list, axis=0).T
    z = np.stack(z_list, axis=0).T
    pval = np.stack(pval_list, axis=0).T

    def my_multitest(pval):
        pval = pval.copy()
        mask = ~np.isnan(pval)
        if np.any(mask):
            pval[mask] = multipletests(pval[mask], method=multitest)[1]
        return pval
    padj = np.apply_along_axis(my_multitest, 1, pval)

    summary = {}
    for i in range(group_onehot.shape[1]):
        summary[labels[i]] = pd.DataFrame({
            "pct_1": pct_1[i],
            "pct_2": pct_2[i],
            "logfc": logfc[i],
            "stat": stat[i],
            "z": z[i],
            "pval": pval[i],
            "padj": padj[i]
        }, index=fnames).sort_values(by=["z"], ascending=False)
    return summary
