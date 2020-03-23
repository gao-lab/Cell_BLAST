import pandas as pd
import numpy as np
import igraph
import operator


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
    
    def prob_reset(self):
        self.graph.vs["raw_prob"] = 0
        self.graph.vs["prop_prob"] = 0  # prob propagated from children
        self.graph.vs["prob"] = 0

    def prob_set(self, hits_cl): #hits_cl is a list of cell ontology classes and pval corresponded to hits
        if hits_cl.shape[0] > 1:
            names = np.unique(hits_cl["cell_ontology_class"])
            for name in names: # 1 - pvalue
                self.get_vertex(name)["raw_prob"] = np.sum(1-hits_cl.loc[hits_cl["cell_ontology_class"] == name, "pval"])/np.sum(1-hits_cl["pval"])
            
    def prob_update(self):
        origins = [v for v in self.graph.vs.select(raw_prob_gt=0)]
        for origin in origins:
            for v in self.graph.bfsiter(origin, mode=igraph.OUT):
                if v != origin:  # bfsiter includes the vertex self
                    v["prop_prob"] += origin["raw_prob"]
        self.graph.vs["prob"] = list(map(
            operator.add, self.graph.vs["raw_prob"],
            self.graph.vs["prop_prob"]
        ))
        
    def cal_longest_paths_to_root(self, weight=None):
        self.graph.vs["longest_paths_to_root"] = float("-Inf")
        roots = self.graph.vs.select(lambda v: v.outdegree() == 0)
        for root in roots:
            root["longest_paths_to_root"]=0
        if weight is None:
            self.graph.es["weight"] = 1.0
        else:
            self.graph.es["weight"] = weight
        for vertex in self.graph.vs[self.graph.topological_sorting(mode = igraph.IN)]:
            for neighbor in self.graph.vs[self.graph.neighborhood(vertex, mode=igraph.IN)]:
                if neighbor != vertex:
                    if neighbor["longest_paths_to_root"] < vertex["longest_paths_to_root"] + \
                    self.graph[neighbor, vertex]:
                        neighbor["longest_paths_to_root"] = vertex["longest_paths_to_root"] + \
                        self.graph[neighbor, vertex]
                    
    def longest_paths_to_root(self, name):
        if type(name) == igraph.Vertex:
            name = name[name.attribute_names()[0]]
        if "longest_paths_to_root" not in self.get_vertex(name).attribute_names():
            self.cal_longest_paths_to_root()
        return self.get_vertex(name)["longest_paths_to_root"]
    
    
def max_prob_leaves(cl_dag, hits_cl, thresh=0.5, min_path=1, retrieve="cell_ontology_class"):
    cl_dag.prob_reset()
    cl_dag.prob_set(hits_cl)
    cl_dag.prob_update() 
    subgraph = cl_dag.graph.subgraph(cl_dag.graph.vs.select(prob_gt=thresh))
    leaves, max_prob = [], 0
    for leaf in subgraph.vs.select(lambda v: v.indegree() == 0):
        if cl_dag.longest_paths_to_root(leaf) >= min_path:
            if leaf["prob"] > max_prob:
                max_prob = leaf["prob"]
                leaves = [leaf[retrieve]]
            elif leaf["prob"] == max_prob:
                leaves.append(leaf[retrieve])
    if len(leaves) == 0:
        return "rejected"
    elif len(leaves) > 1:
        return "ambiguous"
    else:
        return leaves

def cblast_b2c(cl_dag, hits_list, thresh=0.5, min_path=4, retrieve="cell_ontology_class"):
    cblast_blast2co = pd.Series()
    for key in hits_list.keys():
        cblast_blast2co = cblast_blast2co.append(pd.Series(max_prob_leaves(cl_dag, hits_list[key]), index=[key]))  
    return cblast_blast2co


# Metric
def cl_accuracy(cl_dag, source, target, ref_cl_list): #ref_cl_list as a list of unique cl in ref
    if len(source) != len(target):
        return ("False input: different cell number.")
    positive_cl = []
    negative_cl = []
    for query_cl in source.unique():
        positive = False
        for ref_cl in ref_cl_list:
            if cl_dag.is_descendant_of(query_cl, ref_cl): # descendant include query_cl itcl
                positive = True
                break
        if positive == True:    
            positive_cl.append(query_cl) # Note: here query_cl maybe a higher level cl
        else:
            negative_cl.append(query_cl)
    # sensitivity
    accuracy_list={}
    for query_cl in positive_cl:
        mask = source == query_cl
        source_cl = source[mask]
        target_cl = target[mask]
        #cl_accuracy here is sensitivity for this cell type
        cl_accuracy = []
        for i in range(source_cl.shape[0]):
            accuracy = None
            if source_cl[i] == target_cl[i]: 
                accuracy = 1
            elif cl_dag.is_descendant_of(target_cl[i], source_cl[i]):
                accuracy = 1 # descendant results as 1
            elif cl_dag.is_ancestor_of(target_cl[i], source_cl[i]):
                accuracy_ref=[]
                for ref_cl in ref_cl_list:
                    if cl_dag.is_descendant_of(source_cl[i], ref_cl) & cl_dag.is_ancestor_of(target_cl[i], ref_cl):
                        accuracy_ref.append(len(list(cl_dag.graph.bfsiter(cl_dag.get_vertex(ref_cl), mode = "IN")))/ \
                                            len(list(cl_dag.graph.bfsiter(cl_dag.get_vertex(target_cl[i]), mode = "IN"))))
                accuracy = np.mean(accuracy_ref)
            else:
                accuracy = 0
            cl_accuracy.append(accuracy)
        accuracy_list[query_cl] = [sum(mask), (sum(cl_accuracy)/len(source_cl)), True]
    # specificity
    for query_cl in negative_cl:
        mask = source == query_cl
        source_cl = source[mask]
        target_cl = target[mask]
        accuracy_list[query_cl] = [sum(mask), ((sum(target_cl == "rejected") + sum(target_cl == "unassigned"))/len(source_cl)), False]
    return pd.DataFrame(accuracy_list, index=("cell number", "accuracy", "positive")).T

def cl_mba(cl_dag, source, target, ref_cl_list):
    accuracy_list = cl_accuracy(cl_dag=cl_dag, source=source, target=target, ref_cl_list=ref_cl_list)
    balanced_accuracy = np.mean(accuracy_list["accuracy"])
    return balanced_accuracy

# def accuracy(accuracy_list):
#     accuracy = sum(accuracy_list["cell number"]*accuracy_list["accuracy"])/sum(accuracy_list["cell number"])
#     return accuracy
