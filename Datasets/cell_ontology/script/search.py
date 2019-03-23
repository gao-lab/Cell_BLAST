#! /usr/bin/env python
# by caozj
# Sep 5, 2018
# 8:23:49 PM


import sys
import pickle
from fuzzywuzzy import process


with open("./index.pkl", "rb") as f:
    pickled = pickle.load(f)
    search_dict = pickled["search_dict"]

pool = search_dict.keys()
try:
    while True:
        old_hits = set()
        query = input("\033[92mSearch> \033[0m")
        result = process.extract(query, pool, limit=10)
        idx = 1
        for key, score in result:
            if search_dict[key] not in old_hits:
                print("\033[95m[Hit %d] similary = %f\033[0m" % (idx, score))
                old_hits.add(search_dict[key])
                print(search_dict[key].obo[7:])
                print()
                idx += 1
except KeyboardInterrupt:
    print("\nBye!")
    sys.exit(0)
