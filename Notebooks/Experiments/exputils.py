import sys
import numpy as np
import pandas as pd
import scipy.stats
import scipy.linalg
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import scvi.inference
import Cell_BLAST as cb

sys.path.insert(0, "../../Evaluation")
import utils
clean_dataset = utils.clean_dataset
na_mask = utils.na_mask
pick_gpu_lowest_memory = utils.pick_gpu_lowest_memory


def subsample_roc(fpr, tpr, subsample_size=1000):
    dlength = np.concatenate([
        np.zeros(1),
        np.sqrt(np.square(fpr[1:] - fpr[:-1]) + np.square(tpr[1:] - tpr[:-1]))
    ], axis=0)
    length = dlength.sum()
    step = length / subsample_size
    cumlength = dlength.cumsum()
    nstep = np.floor(cumlength / step)
    landmark = np.concatenate([np.zeros(1).astype(np.bool_), nstep[1:] == nstep[:-1]], axis=0)
    return fpr[~landmark], tpr[~landmark]


def get_cb_latent(model, ds, n_posterior=0):
    return model.inference(ds, n_posterior=n_posterior)


def get_cb_log_likelihood(model, ds, latent):
    x = cb.utils.densify(ds.exprs)
    return model.sess.run(model.prob_module.mean_log_likelihood, feed_dict={
        model.x: x,
        model.latent: latent,
        model.training_flag: False
    }) * len(model.genes)


def get_cb_log_prior(model, latent):
    cluster_heads = model._fetch(
        model.sess.graph.get_tensor_by_name("encoder/CatGau/cluster_head/weights/read:0"))
    pdf_list = []
    for cluster_head in cluster_heads:
        pdf_list.append(scipy.stats.multivariate_normal(
            mean=cluster_head).pdf(latent))
    return np.log(np.stack(pdf_list).mean(axis=0))


def get_cb_log_unnormalized_posterior(model, ds, latent):
    return get_cb_log_likelihood(model, ds, latent) + get_cb_log_prior(model, latent)


@torch.no_grad()
def get_scvi_latent(model, ds, n_posterior=0, return_library=False):
    trainer = scvi.inference.UnsupervisedTrainer(model, ds)
    posterior = trainer.create_posterior()
    trainer.model.eval()
    z_list, l_list = [], []
    torch.manual_seed(0)
    for x_, _, _, _, _ in posterior:
        if trainer.model.log_variational:
            x_ = torch.log(1 + x_)
        qz_m, qz_v, z = trainer.model.z_encoder(x_)
        ql_m, ql_v, l = trainer.model.l_encoder(x_)
        if n_posterior > 0:
            z = torch.distributions.Normal(
                qz_m, qz_v.sqrt()
            ).sample(
                torch.Size((n_posterior, ))
            ).transpose(0, 1)
            l = torch.distributions.Normal(
                ql_m, ql_v.sqrt()
            ).sample(
                torch.Size((n_posterior, ))
            )
        else:
            z, l = qz_m, ql_m
        z_list.append(z)
        l_list.append(l)
    if return_library:
        return torch.cat(z_list).cpu().numpy(), torch.cat(l_list).cpu().numpy()
    return torch.cat(z_list).cpu().numpy()


@torch.no_grad()
def get_scvi_log_likelihood(model, ds, latent, library):
    trainer = scvi.inference.UnsupervisedTrainer(model, ds)
    posterior = trainer.create_posterior()
    trainer.model.eval()
    reconst_loss_list = []
    z = torch.from_numpy(latent.astype(np.float32).copy()).cuda()
    l = torch.from_numpy(library.astype(np.float32).copy()).cuda()
    for x_, _, _, _, _ in posterior:
        if trainer.model.log_variational:
            x_ = torch.log(1 + x_)
        z_ = z[:x_.shape[0]]
        z = z[x_.shape[0]:]
        l_ = l[:x_.shape[0]]
        l = l[x_.shape[0]:]
        _, px_r, px_rate, px_dropout = trainer.model.decoder(
            trainer.model.dispersion, z_, l_, None, None)
        if trainer.model.dispersion == "gene":
            px_r = trainer.model.px_r
        else:
            raise NotImplementedError
        px_r = torch.exp(px_r)
        reconst_loss_list.append(
            trainer.model._reconstruction_loss(x_, px_rate, px_r, px_dropout))
    return -torch.cat(reconst_loss_list).cpu().numpy()


def get_scvi_log_prior(ds, latent, library):
    return scipy.stats.multivariate_normal(
        mean=np.zeros(2)
    ).logpdf(latent) + scipy.stats.norm(
        loc=ds.local_means.ravel()[0],
        scale=np.sqrt(ds.local_vars.ravel()[0])
    ).logpdf(library.ravel())


def get_scvi_log_unnormalized_posterior(model, ds, latent, library):
    return get_scvi_log_likelihood(model, ds, latent, library) + get_scvi_log_prior(ds, latent, library)


def metropolis_hastings(
        init, log_unnormalized_posterior, target=100,
        proposal_std=0.02, burnin=100, step=10, random_seed=0
):
    init = [init] if not isinstance(init, list) else init
    proposal_std = [proposal_std] * len(init) if not isinstance(proposal_std, list) else proposal_std
    assert len(init) == len(proposal_std)
    samples = [[] for _ in range(len(init))]
    current = init
    random_state = np.random.RandomState(random_seed)
    for i in cb.utils.smart_tqdm()(range(burnin + target * step)):
        proposal = [
            current_ + random_state.normal(scale=proposal_std_, size=current_.shape)
            for current_, proposal_std_ in zip(current, proposal_std)
        ]
        current_logp = log_unnormalized_posterior(*current)
        proposal_logp = log_unnormalized_posterior(*proposal)
        accept_ratio = np.exp(proposal_logp - current_logp)
        toss = random_state.uniform(size=accept_ratio.shape)
        mask = toss > accept_ratio
        for proposal_, current_ in zip(proposal, current):
            proposal_[mask] = current_[mask]
        if i >= burnin and (i - burnin) % step == 0:
            for samples_, proposal_ in zip(samples, proposal):
                samples_.append(proposal_)
        current = proposal
    return tuple(
        np.stack(samples_, axis=1) for samples_ in samples
    ) if len(samples) > 1 else np.stack(samples_, axis=1)


def aligned_posterior_plot(deviation, lim=None):
    _, _, svd_components = scipy.linalg.svd(deviation, full_matrices=False)

    _, ax = plt.subplots(figsize=(3.5, 3.5))
    ax = sns.kdeplot(
        data=deviation[:, 0], data2=deviation[:, 1],
        shade=True, shade_lowest=False, levels=7, ax=ax
    )
    lim = lim if lim else np.fabs(ax.axis()).max()
    ax.axhline(y=0, c="grey", linewidth=1)
    ax.axvline(x=0, c="grey", linewidth=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("latent 1")
    ax.set_ylabel("latent 2")

    svd_slope = svd_components[:, [1]] / svd_components[:, [0]]
    xendpoint = np.array([-lim, lim])
    yendpoint = svd_slope * xendpoint[np.newaxis]
    ax.plot(xendpoint, yendpoint[0], c="red")
    ax.plot(xendpoint, yendpoint[1], c="blue")

    quater_frac = np.array([
        np.logical_and(deviation[:, 0] < 0, deviation[:, 1] > 0).sum(),
        np.logical_and(deviation[:, 0] > 0, deviation[:, 1] > 0).sum(),
        np.logical_and(deviation[:, 0] < 0, deviation[:, 1] < 0).sum(),
        np.logical_and(deviation[:, 0] > 0, deviation[:, 1] < 0).sum()
    ])
    quater_frac = quater_frac / deviation.shape[0] * 100
    text_kws = dict(
        fontsize=11, horizontalalignment='center', verticalalignment='center',
        transform=ax.transAxes
    )
    ax.text(0.12, 0.94, "%.1f%%" % quater_frac[0], **text_kws)
    ax.text(0.88, 0.94, "%.1f%%" % quater_frac[1], **text_kws)
    ax.text(0.12, 0.06, "%.1f%%" % quater_frac[2], **text_kws)
    ax.text(0.88, 0.06, "%.1f%%" % quater_frac[3], **text_kws)
    return ax


def distance_pair_plot(edist, pdist, correctness):
    df = pd.DataFrame({
        "Euclidean distance": edist,
        "Posterior distance": pdist,
        "Correctness": correctness
    })
    g = sns.JointGrid(x="Euclidean distance",
                      y="Posterior distance", data=df, height=5)
    for _correctness, _df in df.groupby("Correctness"):
        sns.kdeplot(_df["Euclidean distance"],
                    ax=g.ax_marg_x, legend=False, shade=True)
        sns.kdeplot(_df["Posterior distance"], ax=g.ax_marg_y,
                    vertical=True, legend=False, shade=True)
        sns.kdeplot(_df["Euclidean distance"],
                    _df["Posterior distance"], n_levels=10, ax=g.ax_joint)
    ax = sns.scatterplot(
        x="Euclidean distance", y="Posterior distance", hue="Correctness",
        data=df.sample(frac=1, random_state=0), s=5, edgecolor=None, alpha=0.5,
        rasterized=True, ax=g.ax_joint
    )
    ax.set_xlabel(
        f"Euclidean distance (AUC = {sklearn.metrics.roc_auc_score(correctness, -edist):.3f})")
    ax.set_ylabel(
        f"Posterior distance (AUC = {sklearn.metrics.roc_auc_score(correctness, -pdist):.3f})")
    g.ax_joint.legend(frameon=False)
    return ax
