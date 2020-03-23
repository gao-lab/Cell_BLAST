import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import collections
import argparse
import numpy as np
import sklearn.neighbors
import tensorflow as tf
import keras
import dca_modpp.io

import Cell_BLAST as cb
import utils


N_EMPIRICAL = 10000
MAJORITY_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", type=str, required=True)
    parser.add_argument("-r", "--ref", dest="ref", type=str, required=True)
    parser.add_argument("-q", "--query", dest="query", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, required=True)
    parser.add_argument("-a", "--annotation", dest="annotation", type=str, default="cell_ontology_class")
    parser.add_argument("--n-neighbors", dest="n_neighbors", type=int, default=10)
    parser.add_argument("--min-hits", dest="min_hits", type=int, default=2)
    parser.add_argument("-c", "--cutoff", dest="cutoff", type=float, nargs="+", default=[0.1])
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, default=None)
    parser.add_argument("--subsample-ref", dest="subsample_ref", type=int, default=None)
    parser.add_argument("--clean", dest="clean", type=str, default=None)
    cmd_args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = utils.pick_gpu_lowest_memory() \
        if cmd_args.device is None else cmd_args.device
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.set_session(tf.Session(config=config))
    return cmd_args


def main(cmd_args):

    print("Reading data...")
    genes = np.loadtxt(os.path.join(cmd_args.model, "genes.txt"), dtype=np.str)
    ref = cb.data.ExprDataSet.read_dataset(cmd_args.ref)
    ref = utils.clean_dataset(
        ref, cmd_args.clean
    ).to_anndata() if cmd_args.clean else ref.to_anndata()
    ref = ref[np.random.RandomState(cmd_args.seed).choice(
        ref.shape[0], cmd_args.subsample_ref, replace=False
    ), :] if cmd_args.subsample_ref is not None else ref
    ref_label = ref.obs[cmd_args.annotation].values
    ref = dca_modpp.io.normalize(
        ref, genes, filter_min_counts=False, size_factors=10000,
        normalize_input=False, logtrans_input=True
    )
    print("Loading model...")
    os.environ["CUDA_VISIBLE_DEVICES"] = utils.pick_gpu_lowest_memory() \
        if cmd_args.device is None else cmd_args.device
    model = keras.models.load_model(os.path.join(cmd_args.model, "model.h5"))

    print("Projecting to latent space...")
    ref_latent = model.predict({
        "count": ref.X,
        "size_factors": ref.obs.size_factors
    })
    nn = sklearn.neighbors.NearestNeighbors().fit(ref_latent)

    print("Building empirical distribution...")
    np.random.seed(cmd_args.seed)
    idx1 = np.random.choice(ref_latent.shape[0], size=N_EMPIRICAL)
    idx2 = np.random.choice(ref_latent.shape[0], size=N_EMPIRICAL)
    empirical = np.sort(np.sqrt(np.sum(np.square(
        ref_latent[idx1] - ref_latent[idx2]
    ), axis=1)))

    print("Querying...")
    query = cb.data.ExprDataSet.read_dataset(cmd_args.query)
    query = query[:, np.union1d(query.var_names, genes)]
    query = utils.clean_dataset(
        query, cmd_args.clean
    ).to_anndata() if cmd_args.clean else query.to_anndata()
    start_time = time.time()
    query = dca_modpp.io.normalize(
        query, genes, filter_min_counts=False, size_factors=10000,
        normalize_input=False, logtrans_input=True
    )
    query_latent = model.predict({
        "count": query.X,
        "size_factors": query.obs.size_factors
    })
    nnd, nni = nn.kneighbors(query_latent, n_neighbors=cmd_args.n_neighbors)
    pval = np.empty_like(nnd, np.float32)
    time_per_cell = None
    prediction_dict = collections.defaultdict(list)

    for cutoff in cmd_args.cutoff:
        for i in range(nnd.shape[0]):
            for j in range(nnd.shape[1]):
                pval[i, j] = np.searchsorted(empirical, nnd[i, j]) / empirical.size
            uni, count = np.unique(ref_label[
                nni[i][pval[i] < cutoff]
            ], return_counts=True)
            total_count = count.sum()
            if total_count < cmd_args.min_hits:
                prediction_dict[cutoff].append("rejected")
                continue
            argmax = np.argmax(count)
            if count[argmax] / total_count <= MAJORITY_THRESHOLD:
                prediction_dict[cutoff].append("ambiguous")
                continue
            prediction_dict[cutoff].append(uni[argmax])
        prediction_dict[cutoff] = np.array(prediction_dict[cutoff])
        if time_per_cell is None:
            time_per_cell = (
                time.time() - start_time
            ) * 1000 / len(prediction_dict[cutoff])
    print("Time per cell: %.3fms" % time_per_cell)

    print("Saving results...")
    if os.path.exists(cmd_args.output):
        os.remove(cmd_args.output)
    for cutoff in prediction_dict:
        cb.data.write_hybrid_path(prediction_dict[cutoff], "%s//prediction/%s" % (
            cmd_args.output, str(cutoff)
        ))
    cb.data.write_hybrid_path(nni, "//".join((cmd_args.output, "nni")))
    cb.data.write_hybrid_path(nnd, "//".join((cmd_args.output, "nnd")))
    cb.data.write_hybrid_path(pval, "//".join((cmd_args.output, "pval")))
    cb.data.write_hybrid_path(time_per_cell, "//".join((cmd_args.output, "time")))


if __name__ == "__main__":
    main(parse_args())
