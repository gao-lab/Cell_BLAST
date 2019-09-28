import re
import random
import time
import subprocess
import numpy as np
import Cell_BLAST as cb


def clean_dataset(dataset, obs_col):
    mask = na_mask(dataset.obs[obs_col])
    cb.message.info("Cleaning removed %d cells." % mask.sum())
    return dataset[~mask, :]


def na_mask(arr):
    return np.in1d(arr.astype(str), ("", "na", "NA", "nan", "NaN"))


# Adapted from https://stackoverflow.com/questions/41634674/tensorflow-on-shared-gpus-how-to-automatically-select-the-one-that-is-unused
# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

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
