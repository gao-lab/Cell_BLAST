import glob
import Cell_BLAST as cb

for f in glob.glob("../data/*/data.h5"):
    v = cb.data.read_hybrid_path("{f}//var_names".format(f=f))
    print("{ds}: {vhead}".format(
        ds=f.split("/")[2],
        vhead=str(v[0:3])
    ))
