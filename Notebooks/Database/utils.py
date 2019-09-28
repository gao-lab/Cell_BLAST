import os
import re
import time
import random
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def mkdir(fn):
    def wrapped(*args):
        if not os.path.exists(args[1]):
            os.makedirs(args[1])
        return fn(*args)
    return wrapped


@mkdir
def peek(dataset, dataset_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    _ = sns.distplot(dataset.exprs.sum(axis=1), axlabel='nUMI', ax=ax1)
    _ = sns.distplot((dataset.exprs > 0).sum(axis=1), axlabel="nGene", ax=ax2)
    plt.tight_layout()
    fig.savefig("%s/peek.svg" % dataset_name, bbox_inches="tight")


@mkdir
def self_projection(blast, dataset_name):
    hits = blast.query(blast.ref).reconcile_models().filter("pval", 0.05)
    for i in range(len(hits)):  # Remove self-hit (leave one out cv)
        mask = hits.hits[i] == i
        hits.hits[i] = hits.hits[i][~mask]
        hits.dist[i] = hits.dist[i][~mask]
        hits.pval[i] = hits.pval[i][~mask]
    pred = hits.annotate("cell_ontology_class").values.ravel()
    with open("%s/self_projection.txt" % dataset_name, "w") as f:
        covered = ~np.in1d(pred, ["ambiguous", "rejected"])
        print("Coverage = %.4f" % (covered.sum() / covered.size))
        f.write("coverage\t%f\n" % (covered.sum() / covered.size))
        correctness = pred[covered] == blast.ref.obs["cell_ontology_class"][covered]
        print("Accuracy = %.4f" % (correctness.sum() / correctness.size))
        f.write("accuracy\t%f\n" % (correctness.sum() / correctness.size))


def nan_safe(val, f=None):
    try:
        if np.isnan(val):
            return None
    except Exception:
        pass
    if f is not None:
        return f(val)
    return val


DPI = 300


# FIXME: following is duplicated from Evaluation/utils.py


def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    try:
        for line in output.strip().split("\n"):
            m = gpu_regex.match(line)
            assert m, "Couldnt parse "+line
            result.append(int(m.group("gpu_id")))
    except AssertionError:
        pass
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory(rand_sleep=True):
    """Returns GPU with the least allocated memory"""

    if rand_sleep:
        time.sleep(random.uniform(0, 10))
    memory_gpu_map = gpu_memory_map()
    if memory_gpu_map:
        min_memory = min(memory_gpu_map.values())
        best_gpus = [
            gpu for gpu, memory in memory_gpu_map.items()
            if memory == min_memory
        ]
        return str(random.choice(best_gpus))
    else:
        return ""
