"""
Miscellaneous utility functions and classes
"""


import json
import functools
import operator
import collections
import igraph
import numpy as np
import scipy.sparse
import tqdm


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def in_ipynb():  # pragma: no cover
    """
    From a StackOverFlow thread
    """
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True   # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def smart_tqdm():  # pragma: no cover
    if in_ipynb():
        return tqdm.tqdm_notebook
    return tqdm.tqdm


def with_self_graph(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self.graph.as_default():
            return fn(self, *args, **kwargs)
    return wrapped


# Wraps a batch function into minibatch version
def minibatch(batch_size, desc, use_last=False, progress_bar=True):
    def minibatch_wrapper(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            total_size = args[0].shape[0]
            if use_last:
                n_batch = np.ceil(
                    total_size / float(batch_size)
                ).astype(np.int)
            else:
                n_batch = max(1, np.floor(
                    total_size / float(batch_size)
                ).astype(np.int))
            for batch_idx in smart_tqdm()(
                range(n_batch), desc=desc, unit="batches",
                leave=False, disable=not progress_bar
            ):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, total_size)
                this_args = (item[start:end] for item in args)
                func(*this_args, **kwargs)
        return wrapped_func
    return minibatch_wrapper


# Avoid sklearn warning
def encode_integer(label, sort=False):
    label = np.array(label).ravel()
    classes = np.unique(label)
    if sort:
        classes.sort()
    mapping = {v: i for i, v in enumerate(classes)}
    return np.array([mapping[v] for v in label]), classes


# Avoid sklearn warning
def encode_onehot(label, sort=False, ignore=None):
    i, c = encode_integer(label, sort)
    onehot = scipy.sparse.csc_matrix((
        np.ones_like(i, dtype=np.int32), (np.arange(i.size), i)
    ))
    if ignore is None:
        ignore = []
    return onehot[:, ~np.in1d(c, ignore)].tocsr()


class CellTypeDAG(object):

    def __init__(self, graph=None, vdict=None):
        self.graph = igraph.Graph(directed=True) if graph is None else graph
        self.vdict = {} if vdict is None else vdict

    @classmethod
    def load(cls, file):
        if file.endswith(".json"):
            return cls.load_json(file)
        elif file.endswith(".obo"):
            return cls.load_obo(file)
        else:
            raise ValueError("Unexpected file format!")

    @classmethod
    def load_json(cls, file):
        with open(file, "r") as f:
            d = json.load(f)
        dag = cls()
        dag._build_tree(d)
        return dag

    @classmethod
    def load_obo(cls, file):  # Only building on "is_a" relation between CL terms
        import pronto
        ont = pronto.Ontology(file)
        graph, vdict = igraph.Graph(directed=True), {}
        for item in ont:
            if not item.id.startswith("CL"):
                continue
            if "is_obsolete" in item.other and item.other["is_obsolete"][0] == "true":
                continue
            graph.add_vertex(
                name=item.id, cell_ontology_class=item.name,
                desc=str(item.desc), synonyms=[(
                    "%s (%s)" % (syn.desc, syn.scope)
                 ) for syn in item.synonyms]
            )
            assert item.id not in vdict
            vdict[item.id] = item.id
            assert item.name not in vdict
            vdict[item.name] = item.id
            for synonym in item.synonyms:
                if synonym.scope == "EXACT" and synonym.desc != item.name:
                    vdict[synonym.desc] = item.id
        for source in graph.vs:
            for relation in ont[source["name"]].relations:
                if relation.obo_name != "is_a":
                    continue
                for target in ont[source["name"]].relations[relation]:
                    if not target.id.startswith("CL"):
                        continue
                    graph.add_edge(
                        source["name"],
                        graph.vs.find(name=target.id.split()[0])["name"]
                    )
                    # Split because there are many "{is_infered...}" suffix,
                    # falsely joined to the actual id when pronto parses the
                    # obo file
        return cls(graph, vdict)

    def _build_tree(self, d, parent=None):  # For json loading
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

    def get_vertex(self, name):
        return self.graph.vs.find(self.vdict[name])

    def is_related(self, name1, name2):
        return self.is_descendant_of(name1, name2) \
            or self.is_ancestor_of(name1, name2)

    def is_descendant_of(self, name1, name2):
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name1), self.get_vertex(name2)
        )[0][0]
        return np.isfinite(shortest_path)

    def is_ancestor_of(self, name1, name2):
        if name1 not in self.vdict or name2 not in self.vdict:
            return False
        shortest_path = self.graph.shortest_paths(
            self.get_vertex(name2), self.get_vertex(name1)
        )[0][0]
        return np.isfinite(shortest_path)

    def conditional_prob(self, name1, name2):  # p(name1|name2)
        if name1 not in self.vdict or name2 not in self.vdict:
            return 0
        self.graph.vs["prob"] = 0
        v2_parents = list(self.graph.bfsiter(
            self.get_vertex(name2), mode=igraph.OUT))
        v1_parents = list(self.graph.bfsiter(
            self.get_vertex(name1), mode=igraph.OUT))
        for v in v2_parents:
            v["prob"] = 1
        while True:
            changed = False
            for v1_parent in v1_parents[::-1]:  # Reverse may be more efficient
                if v1_parent["prob"] != 0:
                    continue
                v1_parent["prob"] = np.prod([
                    v["prob"] / v.degree(mode=igraph.IN)
                    for v in v1_parent.neighbors(mode=igraph.OUT)
                ])
                if v1_parent["prob"] != 0:
                    changed = True
            if not changed:
                break
        return self.get_vertex(name1)["prob"]

    # def distance_ratio(self, name1, name2, root="cell"):
    #     if name1 not in self.vdict or name2 not in self.vdict:
    #         return 0
    #     v1_parents = set(self.graph.bfsiter(
    #         self.get_vertex(name1), mode=igraph.OUT))
    #     v2_parents = set(self.graph.bfsiter(
    #         self.get_vertex(name2), mode=igraph.OUT))
    #     common_parents = set.intersection(v1_parents, v2_parents)
    #     inf_common_parents = [
    #         parent for parent in common_parents
    #         if not common_parents.intersection(parent.neighbors(mode=igraph.IN))
    #     ]  # If direct children are not common_parents, no descendents are
    #     ratio = 0
    #     for parent in inf_common_parents:
    #         downstream_dist = (
    #             self.graph.shortest_paths(self.get_vertex(name1), parent)[0][0] +
    #             self.graph.shortest_paths(self.get_vertex(name2), parent)[0][0]
    #         ) / 2
    #         upstream_dist = self.graph.shortest_paths(
    #             parent, self.get_vertex(root)
    #         )[0][0]
    #         ratio += upstream_dist / (upstream_dist + downstream_dist)
    #     return (ratio / len(inf_common_parents)) ** 2

    def similarity(self, name1, name2, method="probability"):
        if method == "probability":
            return (
                self.conditional_prob(name1, name2) +
                self.conditional_prob(name2, name1)
            ) / 2
        # if method == "distance":
        #     return self.distance_ratio(name1, name2)
        raise ValueError("Invalid method!")  # pragma: no cover

    def count_reset(self):
        self.graph.vs["raw_count"] = 0
        self.graph.vs["prop_count"] = 0  # count propagated from children
        self.graph.vs["count"] = 0

    def count_set(self, name, count):
        self.get_vertex(name)["raw_count"] = count

    def count_update(self):
        origins = [v for v in self.graph.vs.select(raw_count_gt=0)]
        for origin in origins:
            for v in self.graph.bfsiter(origin, mode=igraph.OUT):
                if v != origin:  # bfsiter includes the vertex self
                    v["prop_count"] += origin["raw_count"]
        self.graph.vs["count"] = list(map(
            operator.add, self.graph.vs["raw_count"],
            self.graph.vs["prop_count"]
        ))

    def best_leaves(self, thresh, retrieve="name"):
        subgraph = self.graph.subgraph(self.graph.vs.select(count_ge=thresh))
        leaves, max_count = [], 0
        for leaf in subgraph.vs.select(lambda v: v.indegree() == 0):
            if leaf["count"] > max_count:
                max_count = leaf["count"]
                leaves = [leaf[retrieve]]
            elif leaf["count"] == max_count:
                leaves.append(leaf[retrieve])
        return leaves


class DataDict(collections.OrderedDict):

    def shuffle(self, random_state=np.random):
        shuffled = DataDict()
        shuffle_idx = None
        for item in self:
            shuffle_idx = random_state.permutation(self[item].shape[0]) \
                if shuffle_idx is None else shuffle_idx
            shuffled[item] = self[item][shuffle_idx]
        return shuffled

    @property
    def size(self):
        data_size = set([item.shape[0] for item in self.values()])
        assert len(data_size) == 1
        return data_size.pop()

    @property
    def shape(self):  # Compatibility with numpy arrays
        return [self.size]

    def __getitem__(self, fetch):
        if isinstance(fetch, (slice, np.ndarray)):
            return DataDict([
                (item, self[item][fetch]) for item in self
            ])
        return super(DataDict, self).__getitem__(fetch)


def densify(arr):
    if scipy.sparse.issparse(arr):
        return arr.toarray()
    return arr


def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
encode = empty_safe(np.vectorize(lambda _x: str(_x).encode("utf-8")), "S")
upper = empty_safe(np.vectorize(lambda x: str(x).upper()), str)
lower = empty_safe(np.vectorize(lambda x: str(x).lower()), str)
tostr = empty_safe(np.vectorize(str), str)
