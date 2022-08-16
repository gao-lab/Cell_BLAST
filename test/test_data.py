import sys
import os
import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import anndata
import matplotlib
matplotlib.use("agg")

if os.environ.get("TEST_MODE", "INSTALL") == "DEV":
    sys.path.insert(0, "..")
import Cell_BLAST as cb


class TestDenseExprDataSet(unittest.TestCase):

    def setUp(self):
        self.X = np.array([
            [2, 3, 0],
            [1, 0, 4],
            [0, 0, 5]
        ])
        self.var = pd.DataFrame({"column": [True, True, False]}, index=["a", "b", "c"])
        self.obs = pd.DataFrame({"column": [1, 1, 1]}, index=["d", "e", "f"])
        self.adata = anndata.AnnData(X=self.X, obs=self.obs, var=self.var)

    def test_normalize(self):
        cb.data.normalize(self.adata, target=100)
        exprs_norm = np.array([
            [40, 60, 0],
            [20, 0, 80],
            [0, 0, 100]
        ])
        self.assertTrue((
            cb.utils.densify(self.adata.X) != exprs_norm
        ).sum() == 0)

    def test_find_variable_genes(self):  # Smoke test
        cb.data.find_variable_genes(self.adata, num_bin=2, grouping="column")

    def test_select_vars(self):
        X_ok = np.array([
            [2, 0, 0],
            [1, 0, 4],
            [0, 0, 5]
        ])
        var_ok = pd.DataFrame({"column": [True, np.nan, False]}, index=["a", "g", "c"])
        obs_ok = pd.DataFrame({"column": [1, 1, 1]}, index=["d", "e", "f"])
        adata_ok = anndata.AnnData(X=X_ok, obs=obs_ok, var=var_ok)
        self._compare_datasets(cb.data.select_vars(self.adata, ["a", "g", "c"]), adata_ok)

    def test_map_vars(self):
        mapping = pd.DataFrame(dict(
            source=["a", "b", "c"],
            target=["A", "B", "C"]
        ))
        mapped_ds = cb.data.map_vars(self.adata, mapping, map_hvg=["column"])
        self.adata.var.index = ["A", "B", "C"]
        self._compare_datasets(self.adata, mapped_ds)

    def _compare_datasets(self, ds1, ds2):
        self.assertTrue(
            (
                cb.utils.densify(ds1.X) !=
                cb.utils.densify(ds2.X)
            ).sum() == 0 and
            np.all(ds1.obs.iloc[~cb.utils.isnan(ds1.obs).ravel(), :].to_numpy() ==
                   ds2.obs.iloc[~cb.utils.isnan(ds2.obs).ravel(), :].to_numpy()) and
            np.all(ds1.var.iloc[~cb.utils.isnan(ds1.var).ravel(), :].to_numpy() ==
                   ds2.var.iloc[~cb.utils.isnan(ds2.var).ravel(), :].to_numpy())
        )


class TestSparseExprDataSet(TestDenseExprDataSet):

    def setUp(self):
        self.X = scipy.sparse.csr_matrix(np.array([
            [2, 3, 0],
            [1, 0, 4],
            [0, 0, 5]
        ]))
        self.var = pd.DataFrame({"column": [True, True, False]}, index=["a", "b", "c"])
        self.obs = pd.DataFrame({"column": [1, 1, 1]}, index=["d", "e", "f"])
        self.adata = anndata.AnnData(X=self.X, obs=self.obs, var=self.var)


if __name__ == "__main__":
    # TestDenseExprDataSet().setUpClass()
    # t = TestDenseExprDataSet()
    # t.setUp()
    # t.test_select_vars()
    # t.tearDown()
    # TestDenseExprDataSet().tearDownClass()
    unittest.main()
