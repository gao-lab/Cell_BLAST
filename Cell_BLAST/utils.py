r"""
Miscellaneous utility functions and classes
"""

import binascii
import collections
import functools
import json
import logging
import operator
import os
import re
import typing

import h5py
import igraph
import numpy as np
import pandas as pd
import pynvml
import scipy.sparse
import torch
import tqdm
import tqdm.notebook

log_handler = logging.StreamHandler()
log_handler.setLevel(logging.INFO)
log_handler.setFormatter(logging.Formatter(fmt="[%(levelname)s] %(name)s: %(message)s"))
logger = logging.getLogger("Cell BLAST")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)


def rand_hex() -> str:
    return binascii.b2a_hex(os.urandom(15)).decode()


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def in_ipynb() -> bool:  # pragma: no cover
    r"""
    From a StackOverFlow thread
    """
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def smart_tqdm():  # pragma: no cover
    if in_ipynb():
        return tqdm.notebook.tqdm
    return tqdm.tqdm


def with_self_graph(fn: typing.Callable) -> typing.Callable:
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self.graph.as_default():
            return fn(self, *args, **kwargs)

    return wrapped


# Wraps a batch function into minibatch version
def minibatch(
    batch_size: int, desc: str, use_last: bool = False, progress_bar: bool = True
) -> typing.Callable:
    def minibatch_wrapper(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            total_size = args[0].shape[0]
            if use_last:
                n_batch = np.ceil(total_size / float(batch_size)).astype(np.int)
            else:
                n_batch = max(
                    1, np.floor(total_size / float(batch_size)).astype(np.int)
                )
            for batch_idx in smart_tqdm()(
                range(n_batch),
                desc=desc,
                unit="batches",
                leave=False,
                disable=not progress_bar,
            ):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, total_size)
                this_args = (item[start:end] for item in args)
                func(*this_args, **kwargs)

        return wrapped_func

    return minibatch_wrapper


# Avoid sklearn warning
def encode_integer(
    label: typing.List[typing.Any], sort: bool = False
) -> typing.Tuple[np.ndarray, np.ndarray]:
    index = pd.Index(label).dropna().drop_duplicates()
    if sort:
        index = index.sort_values()
    return index.get_indexer(label), index.to_numpy()


# Avoid sklearn warning
def encode_onehot(
    label: typing.List[typing.Any], sort: bool = False
) -> scipy.sparse.csr_matrix:
    i, c = encode_integer(label, sort)
    val = np.ones_like(i, dtype=np.int32)[i >= 0]
    row = np.arange(i.size)[i >= 0]
    col = i[i >= 0]
    return scipy.sparse.csr_matrix((val, (row, col)), shape=(i.size, c.size))


class CellTypeDAG(object):
    def __init__(
        self,
        graph: typing.Optional[igraph.Graph] = None,
        vdict: typing.Optional[typing.Mapping[str, str]] = None,
    ) -> None:
        self.graph = igraph.Graph(directed=True) if graph is None else graph
        self.vdict = vdict or {}

    @classmethod
    def load(cls, file: str) -> "CellTypeDAG":
        if file.endswith(".json"):
            return cls.load_json(file)
        elif file.endswith(".obo"):
            return cls.load_obo(file)
        else:
            raise ValueError("Unexpected file format!")

    @classmethod
    def load_json(cls, file: str) -> "CellTypeDAG":
        with open(file, "r") as f:
            d = json.load(f)
        dag = cls()
        dag._build_tree(d)
        return dag

    @classmethod
    def load_obo(
        cls, file: str
    ) -> "CellTypeDAG":  # Only building on "is_a" relation between CL terms
        import pronto

        ont = pronto.Ontology(file)
        graph, vdict = igraph.Graph(directed=True), {}
        for item in ont:
            if not item.id.startswith("CL"):
                continue
            if "is_obsolete" in item.other and item.other["is_obsolete"][0] == "true":
                continue
            graph.add_vertex(
                name=item.id,
                cell_ontology_class=item.name,
                desc=str(item.desc),
                synonyms=[(f"{syn.desc} ({syn.scope})") for syn in item.synonyms],
            )
            assert item.id not in vdict
            vdict[item.id] = item.id
            assert item.name not in vdict
            vdict[item.name] = item.id
            for synonym in item.synonyms:
                if synonym.scope == "EXACT" and synonym.desc != item.name:
                    vdict[synonym.desc] = item.id
        for source in graph.vs:  # pylint: disable=not-an-iterable
            for relation in ont[source["name"]].relations:
                if relation.obo_name != "is_a":
                    continue
                for target in ont[source["name"]].relations[relation]:
                    if not target.id.startswith("CL"):
                        continue
                    graph.add_edge(
                        source["name"],
                        graph.vs.find(name=target.id.split()[0])[
                            "name"
                        ],  # pylint: disable=no-member
                    )
                    # Split because there are many "{is_infered...}" suffix,
                    # falsely joined to the actual id when pronto parses the
                    # obo file
        return cls(graph, vdict)

    def _build_tree(
        self, d: typing.Mapping[str, str], parent: typing.Optional[igraph.Vertex] = None
    ) -> None:  # For json loading
        self.graph.add_vertex(name=d["name"])
        v = self.graph.vs.find(d["name"])
        if parent is not None:
            self.graph.add_edge(v, parent)
        self.vdict[d["name"]] = d["name"]
        if "alias" in d:
            for alias in d["alias"]:
                self.vdict[alias] = d["name"]
        if "children" in d:
            for subd in d["children"]:
                self._build_tree(subd, v)

    def get_vertex(self, name: str) -> igraph.Vertex:
        return self.graph.vs.find(self.vdict[name])

    def is_related(self, name1: str, name2: str) -> bool:
        return self.is_descendant_of(name1, name2) or self.is_ancestor_of(name1, name2)

    def is_descendant_of(self, name1: str, name2: str) -> bool:
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name1), self.get_vertex(name2)
        )[0][0]
        return np.isfinite(shortest_path)

    def is_ancestor_of(self, name1: str, name2: str) -> bool:
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name2), self.get_vertex(name1)
        )[0][0]
        return np.isfinite(shortest_path)

    def conditional_prob(self, name1: str, name2: str) -> float:  # p(name1|name2)
        if name1 not in self.vdict or name2 not in self.vdict:
            return 0
        self.graph.vs["prob"] = 0
        v2_parents = list(self.graph.bfsiter(self.get_vertex(name2), mode=igraph.OUT))
        v1_parents = list(self.graph.bfsiter(self.get_vertex(name1), mode=igraph.OUT))
        for v in v2_parents:
            v["prob"] = 1
        while True:
            changed = False
            for v1_parent in v1_parents[::-1]:  # Reverse may be more efficient
                if v1_parent["prob"] != 0:
                    continue
                v1_parent["prob"] = np.prod(
                    [
                        v["prob"] / v.degree(mode=igraph.IN)
                        for v in v1_parent.neighbors(mode=igraph.OUT)
                    ]
                )
                if v1_parent["prob"] != 0:
                    changed = True
            if not changed:
                break
        return self.get_vertex(name1)["prob"]

    def similarity(self, name1: str, name2: str, method: str = "probability") -> float:
        if method == "probability":
            return (
                self.conditional_prob(name1, name2)
                + self.conditional_prob(name2, name1)
            ) / 2
        # if method == "distance":
        #     return self.distance_ratio(name1, name2)
        raise ValueError("Invalid method!")  # pragma: no cover

    def value_reset(self) -> None:
        self.graph.vs["raw_value"] = 0
        self.graph.vs["prop_value"] = 0  # value propagated from children
        self.graph.vs["value"] = 0

    def value_set(self, name: str, value: float) -> None:
        try:
            self.get_vertex(name)["raw_value"] = value
        except KeyError:
            logger.warning("Unknown node name! Doing nothing.")

    def value_update(self) -> None:
        origins = [v for v in self.graph.vs.select(raw_value_gt=0)]
        for origin in origins:
            for v in self.graph.bfsiter(origin, mode=igraph.OUT):
                if v != origin:  # bfsiter includes the vertex self
                    v["prop_value"] += origin["raw_value"]
        self.graph.vs["value"] = list(
            map(operator.add, self.graph.vs["raw_value"], self.graph.vs["prop_value"])
        )

    def best_leaves(
        self, thresh: float = 0.5, min_path: int = 4, retrieve: str = "name"
    ) -> typing.List[str]:
        subgraph = self.graph.subgraph(self.graph.vs.select(value_gt=thresh))
        leaves, max_value = [], 0
        for leaf in subgraph.vs.select(lambda v: v.indegree() == 0):
            if self.longest_paths_to_root(leaf["name"]) < min_path:
                continue
            if leaf["value"] > max_value:
                max_value = leaf["value"]
                leaves = [leaf[retrieve]]
            elif leaf["value"] == max_value:
                leaves.append(leaf[retrieve])
        return leaves

    def cal_longest_paths_to_root(self, weight: float = 1.0) -> None:
        self.graph.vs["longest_paths_to_root"] = -np.inf
        roots = self.graph.vs.select(lambda v: v.outdegree() == 0)
        for root in roots:
            root["longest_paths_to_root"] = 0
        self.graph.es["weight"] = weight
        for vertex in self.graph.vs[self.graph.topological_sorting(mode=igraph.IN)]:
            for neighbor in self.graph.vs[
                self.graph.neighborhood(vertex, mode=igraph.IN)
            ]:
                if neighbor == vertex:
                    continue
                if (
                    neighbor["longest_paths_to_root"]
                    < vertex["longest_paths_to_root"] + self.graph[neighbor, vertex]
                ):
                    neighbor["longest_paths_to_root"] = (
                        vertex["longest_paths_to_root"] + self.graph[neighbor, vertex]
                    )

    def longest_paths_to_root(self, name: str) -> int:
        if "longest_paths_to_root" not in self.get_vertex(name).attribute_names():
            self.cal_longest_paths_to_root()
        return self.get_vertex(name)["longest_paths_to_root"]


class DataDict(collections.OrderedDict):
    def shuffle(self, random_state: np.random.RandomState = np.random) -> "DataDict":
        shuffled = DataDict()
        shuffle_idx = None
        for item in self:
            shuffle_idx = (
                random_state.permutation(self[item].shape[0])
                if shuffle_idx is None
                else shuffle_idx
            )
            shuffled[item] = self[item][shuffle_idx]
        return shuffled

    @property
    def size(self) -> int:
        data_size = set([item.shape[0] for item in self.values()])
        if data_size:
            assert len(data_size) == 1
            return data_size.pop()
        return 0

    @property
    def shape(self) -> typing.List[int]:  # Compatibility with numpy arrays
        return [self.size]

    def __getitem__(
        self, fetch: typing.Union[str, slice, np.ndarray]
    ) -> typing.Union["DataDict", np.ndarray]:
        if isinstance(fetch, (slice, np.ndarray)):
            return DataDict([(item, self[item][fetch]) for item in self])
        return super(DataDict, self).__getitem__(fetch)


def densify(arr: typing.Union[np.ndarray, scipy.sparse.spmatrix]) -> np.ndarray:
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn: typing.Callable, dtype: type):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)

    return _fn


def scope_free(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\\-]", "_", x)


def isnan(x: typing.Any) -> bool:
    try:
        return np.isnan(x)
    except Exception:
        return False


isnan = empty_safe(np.vectorize(isnan), bool)
decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)


def read_hybrid_path(hybrid_path: str) -> np.ndarray:
    file_name, h5_path = hybrid_path.split("//")
    with h5py.File(file_name, "r") as f:
        obj = f[h5_path][...]
    if obj.dtype.type is np.bytes_:
        obj = decode(obj)
    return obj


def write_hybrid_path(obj: np.ndarray, hybrid_path: str) -> None:
    file_name, h5_path = hybrid_path.split("//")
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with h5py.File(file_name, "a") as f:
        if h5_path in f:
            del f[h5_path]
        if obj.dtype.type in (np.str_, np.object_):
            obj = encode(obj)
        f.create_dataset(h5_path, data=obj)


def dataframe2list(dataframe):
    identical = []
    for i in zip(dataframe.iloc[:, 0], dataframe.iloc[:, 1]):
        identical.append(i)
    return identical


@functools.lru_cache(maxsize=1)
def autodevice() -> torch.device:
    r"""
    Get torch computation device automatically
    based on GPU availability and memory usage

    Returns
    -------
    device
        Computation device
    """
    used_device = -1
    try:
        pynvml.nvmlInit()
        free_mems = np.array(
            [
                pynvml.nvmlDeviceGetMemoryInfo(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                ).free
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
        )
        if free_mems.size:
            best_devices = np.where(free_mems == free_mems.max())[0]
            used_device = np.random.choice(best_devices, 1)[0]
    except pynvml.NVMLError:
        pass
    if used_device == -1:
        logger.info("Using CPU as computation device.")
        return torch.device("cpu")
    logger.info("Using GPU %d as computation device.", used_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(used_device)
    return torch.device("cuda")
