import sys
import os
import unittest
import collections
import numpy as np
import pandas as pd
import scipy.sparse
import anndata
import matplotlib
matplotlib.use("agg")

if os.environ.get("TEST_MODE", "INSTALL") == "DEV":
    sys.path.insert(0, "..")
import Cell_BLAST as cb


class TestHybridPath(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m1 = np.array([
            [2, 1, 0],
            [3, 0, 0],
            [0, 4, 5]
        ])
        cls.v2 = np.array(["a", "s", "d"])
        cls.d3 = {
            "m1": cls.m1,
            "v2": cls.v2
        }
        cls.s4 = "asd"

    def test_hybrid_path(self):
        cb.data.write_hybrid_path(self.m1, "./test.h5//a")
        cb.data.write_hybrid_path(self.v2, "./test.h5//b/c")
        cb.data.write_hybrid_path(self.d3, "./test.h5//b/d/e")
        cb.data.write_hybrid_path(self.s4, "./test.h5//f")
        self.assertTrue(cb.data.check_hybrid_path("./test.h5//b/c"))
        self.assertFalse(cb.data.check_hybrid_path("./test.h5//b/f"))
        self.assertFalse(cb.data.check_hybrid_path("./asd.h5//b/f"))
        m1 = cb.data.read_hybrid_path("./test.h5//a")
        v2 = cb.data.read_hybrid_path("./test.h5//b/c")
        d3 = cb.data.read_hybrid_path("./test.h5//b/d/e")
        s4 = cb.data.read_hybrid_path("./test.h5//f")
        self.assertTrue(np.all(self.m1 == m1))
        self.assertTrue(np.all(self.v2 == v2))
        self.assertTrue(
            np.all(self.d3["m1"] == d3["m1"]) and
            np.all(self.d3["v2"] == d3["v2"])
        )
        self.assertEqual(self.s4, s4)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("./test.h5"):
            os.remove("./test.h5")


class TestDenseExprDataSet(unittest.TestCase):

    def setUp(self):
        self.exprs = np.array([
            [2, 3, 0],
            [1, 0, 4],
            [0, 0, 5]
        ])
        self.var = pd.DataFrame({"column": [1, 2, 3]}, index=["a", "b", "c"])
        self.obs = pd.DataFrame({"column": [1, 1, 1]}, index=["d", "e", "f"])
        self.uns = {"item": np.array(["a", "c"]), "blah": np.array(["blah"])}
        self.ds = cb.data.ExprDataSet(
            exprs=self.exprs, obs=self.obs, var=self.var, uns=self.uns)

    def test_constructor(self):
        with self.assertRaises(AssertionError):
            _ = cb.data.ExprDataSet(
                exprs=self.exprs, obs=self.obs[1:], var=self.var, uns=self.uns)
        with self.assertRaises(AssertionError):
            _ = cb.data.ExprDataSet(
                exprs=self.exprs, obs=self.obs, var=self.var[1:], uns=self.uns)

    def test_attributes(self):
        self.assertTrue(np.all(self.ds.var_names == np.array(["a", "b", "c"])))
        self.assertTrue(np.all(self.ds.obs_names == np.array(["d", "e", "f"])))
        self.assertTrue(np.all(self.ds.shape == np.array([3, 3])))

    def test_copy(self):
        ds = self.ds.copy(deep=True)
        ds.var_names = ["A", "B", "C"]
        ds.obs_names = ["D", "E", "F"]
        ds.X[0, 0] = 123
        self.assertTrue(np.all(ds.var_names == np.array(["A", "B", "C"])))
        self.assertTrue(np.all(ds.obs_names == np.array(["D", "E", "F"])))
        self.assertEqual(ds.exprs[0, 0], 123)
        self.assertTrue(np.all(self.ds.var_names == np.array(["a", "b", "c"])))
        self.assertTrue(np.all(self.ds.obs_names == np.array(["d", "e", "f"])))
        self.assertEqual(self.ds.exprs[0, 0], 2)

    def test_read_and_write(self):
        self.ds.write_dataset("./test.h5")
        ds = cb.data.ExprDataSet.read_dataset("./test.h5")
        self._compare_datasets(ds, self.ds)

    def test_read_and_write_table(self):
        self.ds.write_table("./test.txt", orientation="gc")
        ds = cb.data.ExprDataSet.read_table("./test.txt", orientation="gc", index_col=0)
        self.assertAlmostEqual(np.max(np.abs(
            cb.utils.densify(self.ds.exprs) - ds.exprs
        )), 0)

        self.ds.write_table("./test.txt", orientation="cg")
        ds = cb.data.ExprDataSet.read_table("./test.txt", orientation="cg", index_col=0)
        self.assertAlmostEqual(np.max(np.abs(
            cb.utils.densify(self.ds.exprs) - ds.exprs
        )), 0)

    def test_loom(self):
        with self.ds.to_loom("./test.loom") as lm:
            ds = cb.data.ExprDataSet.from_loom(lm)
        self.ds.uns = {}
        self.assertTrue(np.all(
            cb.utils.densify(self.ds.exprs) == cb.utils.densify(ds.exprs)))

    def test_normalize(self):
        ds = self.ds.normalize(target=100)
        exprs_norm = np.array([
            [40, 60, 0],
            [20, 0, 80],
            [0, 0, 100]
        ])
        self.assertTrue((
            cb.utils.densify(ds.exprs) != exprs_norm
        ).sum() == 0)

    def test_select_genes(self):  # Smoke test
        self.ds.find_variable_genes(num_bin=2, grouping="column")

    def test_slicing(self):
        exprs_ok = np.array([
            [2, 0, 0],
            [0, 0, 5]
        ])
        var_ok = pd.DataFrame({"column": [1, np.nan, 3]}, index=["a", "g", "c"])
        obs_ok = pd.DataFrame({"column": [1, 1]}, index=["d", "f"])
        ds_ok = cb.data.ExprDataSet(
            exprs=exprs_ok, obs=obs_ok, var=var_ok, uns=self.ds.uns)
        self._compare_datasets(self.ds[["d", "f"], ["a", "g", "c"]], ds_ok)

        exprs_ok = np.array([
            [2, 0],
            [0, 5]
        ])
        var_ok = pd.DataFrame({"column": [1, 3]}, index=["a", "c"])
        obs_ok = pd.DataFrame({"column": [1, 1]}, index=["d", "f"])
        ds_ok = cb.data.ExprDataSet(
            exprs=exprs_ok, obs=obs_ok, var=var_ok, uns=self.ds.uns)
        self._compare_datasets(self.ds[:, :][[0, 2], [0, 2]], ds_ok)
        self._compare_datasets(self.ds[
            [True, False, True], [True, False, True]
        ], ds_ok)
        _ = self.ds["d", "a"]
        _ = self.ds[1, 2]
        _ = self.ds[[], []]

    def test_map_vars(self):
        mapping = pd.DataFrame(dict(
            source=["a", "b", "c"],
            target=["A", "B", "C"]
        ))
        mapped_ds = self.ds.map_vars(mapping)
        mapped_ds = self.ds.map_vars(mapping, map_uns_slots=["item"])
        mapped_ds.var["column"] = np.arange(3) + 1
        self.ds.var.index = ["A", "B", "C"]
        self.ds.uns["item"] = ["A", "C"]
        self._compare_datasets(self.ds, mapped_ds)

    def test_merge_datasets(self):
        merged_ds = cb.data.ExprDataSet.merge_datasets(collections.OrderedDict(
            ds1=self.ds, ds2=self.ds.copy(deep=True)
        ), meta_col="study", merge_uns_slots=["item"])
        exprs_ok = cb.utils.densify(self.ds.exprs)
        exprs_ok = np.concatenate([exprs_ok, exprs_ok], axis=0)
        obs_ok = pd.DataFrame(collections.OrderedDict(
            column=np.concatenate([
                self.ds.obs["column"].values,
                self.ds.obs["column"].values
            ], axis=0),
            study=np.concatenate([
                np.repeat("ds1", 3),
                np.repeat("ds2", 3)
            ], axis=0)
        ), index=["d", "e", "f"] * 2)
        var_ok = pd.DataFrame(collections.OrderedDict(
            column_ds1=np.arange(3) + 1,
            column_ds2=np.arange(3) + 1
        ), index=["a", "b", "c"])
        uns_ok = np.array(["a", "c"])
        self.assertTrue(
            np.sum(cb.utils.densify(merged_ds.exprs) != exprs_ok) == 0 and
            np.all(np.all(merged_ds.obs == obs_ok)) and
            np.all(np.all(merged_ds.var == var_ok)) and
            np.all(merged_ds.uns["item"] == uns_ok)
        )

    def test_anndata(self):
        ad = self.ds.to_anndata()
        ad.write_h5ad("./test.h5ad")
        ds = cb.data.ExprDataSet.from_anndata(anndata.read_h5ad("./test.h5ad"))
        self._compare_datasets(self.ds, ds)

    def _compare_datasets(self, ds1, ds2):
        self.assertTrue(
            (
                cb.utils.densify(ds1.exprs) !=
                cb.utils.densify(ds2.exprs)
            ).sum() == 0 and
            np.all(ds1.obs.iloc[~np.isnan(ds1.obs).values.ravel(), :].values ==
                   ds2.obs.iloc[~np.isnan(ds2.obs).values.ravel(), :].values) and
            np.all(ds1.var.iloc[~np.isnan(ds1.var).values.ravel(), :].values ==
                   ds2.var.iloc[~np.isnan(ds2.var).values.ravel(), :].values) and
            np.all(ds1.uns["item"] == ds2.uns["item"])
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("./test.h5"):
            os.remove("./test.h5")
        if os.path.exists("./test.h5ad"):
            os.remove("./test.h5ad")
        if os.path.exists("./test.txt"):
            os.remove("./test.txt")
        if os.path.exists("./test.loom"):
            os.remove("./test.loom")


class TestSparseExprDataSet(TestDenseExprDataSet):

    def setUp(self):
        self.exprs = scipy.sparse.csr_matrix(np.array([
            [2, 3, 0],
            [1, 0, 4],
            [0, 0, 5]
        ]))
        self.var = pd.DataFrame({"column": [1, 2, 3]}, index=["a", "b", "c"])
        self.obs = pd.DataFrame({"column": [1, 1, 1]}, index=["d", "e", "f"])
        self.uns = {"item": np.array(["a", "c"])}
        self.ds = cb.data.ExprDataSet(
            exprs=self.exprs, obs=self.obs, var=self.var, uns=self.uns)


class TestOtherDatasetUtilities(unittest.TestCase):

    def setUp(self):
        self.ds = cb.data.ExprDataSet.read_dataset(
            "./pollen.h5", sparsify=True
        ).normalize()

    def test_corr_heatmap(self):
        _ = self.ds.obs_correlation_heatmap(group="cell_type1")

    def test_violin(self):
        _ = self.ds.violin("cell_type1", "A1BG")

    def test_marker(self):
        _ = self.ds.fast_markers("cell_type1", self.ds.uns["scmap_genes"])


if __name__ == "__main__":
    # test = TestDenseExprDataSet()
    # test.setUp()
    # test.test_loom()
    # test.tearDown()
    unittest.main()
