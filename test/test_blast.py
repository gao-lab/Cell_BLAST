import sys
import os
import shutil
import unittest
import numpy as np
import matplotlib
matplotlib.use("agg")

if os.environ.get("TEST_MODE", "INSTALL") == "DEV":
    sys.path.insert(0, "..")
import Cell_BLAST as cb
cb.config.RANDOM_SEED = 0
cb.config.N_JOBS = 2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class BLASTTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = cb.data.ExprDataSet.read_dataset(
            "pollen.h5"
        ).normalize()
        cls.models = []
        for i in range(2):
            cls.models.append(cb.directi.fit_DIRECTi(
                cls.data, cls.data.uns["scmap_genes"], supervision="cell_type1",
                latent_dim=10, epoch=3, random_seed=i, path="test_directi"
            ))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("./test_directi"):
            shutil.rmtree("./test_directi")

    def tearDown(self):
        if os.path.exists("./test_blast"):
            shutil.rmtree("./test_blast")
        if os.path.exists("./tmp"):
            shutil.rmtree("./tmp")

    def test_npdv1_dist_blast(self):
        blast = cb.blast.BLAST(self.models, self.data, distance_metric="npd_v1", n_posterior=20, force_components=False)
        hits = blast[:].query(self.data)
        self.assertEqual(len(hits), self.data.shape[0])
        prediction = hits.reconcile_models(dist_method="min").filter(by="dist").annotate("cell_type1", return_evidence=True).loc[:, ["cell_type1", "n_hits"]]
        _ = hits.reconcile_models().to_data_frames()
        blast.save("./test_blast")
        blast2 = cb.blast.BLAST.load("./test_blast")
        hits2 = blast2.query(self.data)
        prediction2 = hits2.reconcile_models(dist_method="min").filter(by="dist").annotate("cell_type1", return_evidence=True).loc[:, ["cell_type1", "n_hits"]]
        self.assertTrue(np.all(prediction.values == prediction2.values))

    def test_euclidean_dist_blast(self):
        blast = cb.blast.BLAST(self.models, self.data, distance_metric="ed", cluster_empirical=True)
        hits = blast.query(self.data)
        prediction = hits.reconcile_models(dist_method="max").filter(by="dist").annotate("cell_type1")
        blast.save("./test_blast")
        blast2 = cb.blast.BLAST.load("./test_blast")
        hits2 = blast2.query(self.data)
        prediction2 = hits2.reconcile_models(dist_method="max").filter(by="dist").annotate("cell_type1")
        self.assertTrue(np.all(prediction.values == prediction2.values))
        _ = hits.filter(by="dist")[0]
        _ = hits.reconcile_models().filter(by="dist")[0:3]

    def test_amd_pval_blast(self):
        blast = cb.blast.BLAST(self.models, self.data, distance_metric="amd", n_posterior=20, cluster_empirical=True, force_components=False)
        hits = blast.query(self.data, store_dataset=True)
        prediction = hits.filter(by="pval", cutoff=0.05).annotate("cell_type1")
        blast.save("./test_blast")
        blast2 = cb.blast.BLAST.load("./test_blast")
        hits2 = blast2.query(self.data)
        prediction2 = hits2.filter(by="pval", cutoff=0.05).annotate("cell_type1")
        self.assertTrue(np.all(prediction.values == prediction2.values))
        _ = hits.filter(by="pval", cutoff=0.05)[0]
        _ = hits.reconcile_models().filter(by="pval", cutoff=0.05)[0:3]

    def test_npdv2_pval_blast(self):
        blast = cb.blast.BLAST(self.models, self.data, distance_metric="npd_v2", n_posterior=20)
        hits = blast.query(self.data)
        for _ in hits:
            pass
        prediction = hits.reconcile_models().filter(by="pval").annotate("ABRACL").fillna(0)
        blast.save("./test_blast")
        blast2 = cb.blast.BLAST.load("./test_blast")
        hits2 = blast2.query(self.data, store_dataset=True)
        prediction2 = hits2.reconcile_models().filter(by="pval").annotate("ABRACL").fillna(0)
        self.assertTrue(np.all(prediction.values == prediction2.values))

    def test_align_blast(self):
        blast = cb.blast.BLAST(self.models, self.data, distance_metric="md", n_posterior=20, force_components=False)
        blast = blast.align(dict(query=self.data), path="tmp", epoch=3, patience=3)
        hits = blast.query(self.data)
        prediction = hits.reconcile_models().filter(by="pval").annotate("cell_type1")
        blast.save("./test_blast")
        blast2 = cb.blast.BLAST.load("./test_blast", mode=cb.blast.MINIMAL)
        hits2 = blast2.query(self.data, store_dataset=True)
        prediction2 = hits2.reconcile_models().filter(by="pval").annotate("cell_type1")
        self.assertTrue(np.all(prediction.values == prediction2.values))


if __name__ == "__main__":
    # BLASTTest.setUpClass()
    # test = BLASTTest()
    # test.setUp()
    # test.test_npdv1_dist_blast()
    # test.tearDown()
    unittest.main()
