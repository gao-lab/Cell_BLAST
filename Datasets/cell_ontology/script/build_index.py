#! /usr/bin/env python
# by caozj
# Sep 5, 2018
# 8:11:29 PM


import pickle
import pronto


ont = pronto.Ontology("../CL_repo/cl.obo")
search_dict = {}
for term in ont:
    if not term.id.startswith("CL"):
        continue
    search_dict[term.id] = term
    search_dict[term.name] = term
    search_dict[term.desc] = term
    for synonym in term.synonyms:
        search_dict[synonym] = term

with open("./index.pkl", "wb") as f:
    pickle.dump({
        "search_dict": search_dict,
        "ont": ont
    }, f)

print("Done!")
