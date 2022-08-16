# Data Migration from h5 format to h5ad(anndata) format

Since the new version of Cell BLAST only support h5ad(anndata) format of data, we provide a function and a script to help to transform the original h5 format data to h5ad(anndata) format.

The function can be called as following:

```py
import Cell_BLAST as cb
cb.data.h5_to_h5ad(inputfilename, outputfilename)
```

Or, you can run the script we provided:

```sh
python h5_to_h5ad.py -i <inputfilename> -o <outputfilename>
```