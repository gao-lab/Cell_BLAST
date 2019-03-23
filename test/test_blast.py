import sys
import os
import shutil
import unittest
import numpy as np
import matplotlib
matplotlib.use("agg")

sys.path.insert(0, "..")
import Cell_BLAST as cb
cb.config.RANDOM_SEED = 0
cb.config.N_JOBS = 2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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

    def test_posterior_dist_blast(self):
        blast = cb.blast.BLAST(self.models, self.data, n_posterior=20)
        hits = blast[:].query(self.data)
        self.assertEqual(len(hits), self.data.shape[0])
        prediction = hits.reconcile_models(dist_method="min").filter(by="dist").annotate("cell_type1")
        _ = hits.reconcile_models().to_data_frames()
        blast.save("./test_blast")
        blast2 = cb.blast.BLAST.load("./test_blast")
        hits2 = blast2.query(self.data)
        prediction2 = hits2.reconcile_models(dist_method="min").filter(by="dist").annotate("cell_type1")
        self.assertTrue(np.all(prediction.values == prediction2.values))

    def test_euclidean_dist_blast(self):
        blast = cb.blast.BLAST(self.models, self.data, n_posterior=0)
        hits = blast.query(self.data)
        prediction = hits.reconcile_models(dist_method="max").filter(by="dist").annotate("cell_type1")
        blast.save("./test_blast")
        blast2 = cb.blast.BLAST.load("./test_blast")
        hits2 = blast2.query(self.data)
        prediction2 = hits2.reconcile_models(dist_method="max").filter(by="dist").annotate("cell_type1")
        self.assertTrue(np.all(prediction.values == prediction2.values))
        print(hits.filter(by="dist")[0:3])
        print(hits.reconcile_models().filter(by="dist")[0:3])

    def test_posterior_pval_blast(self):
        blast = cb.blast.BLAST(
            self.models, self.data, n_posterior=20
        ).build_empirical(background=self.data)
        hits = blast.query(self.data)
        prediction = hits.reconcile_models().filter(by="pval", cutoff=0.05).annotate("cell_type1")
        blast.save("./test_blast")
        blast2 = cb.blast.BLAST.load("./test_blast")
        hits2 = blast2.query(self.data)
        prediction2 = hits2.reconcile_models().filter(by="pval", cutoff=0.05).annotate("cell_type1")
        self.assertTrue(np.all(prediction.values == prediction2.values))
        print(hits.filter(by="pval", cutoff=0.05)[0:3])
        print(hits.reconcile_models().filter(by="pval", cutoff=0.05)[0:3])

    def test_euclidean_pval_blast(self):
        blast = cb.blast.BLAST(
            self.models, self.data, keep_exprs=True, n_posterior=0
        ).build_empirical(background=self.data)
        hits = blast.query(self.data)
        prediction = hits.reconcile_models().filter(by="pval").annotate(["A1BG", "ZZZ3"]).fillna(0)
        blast.save("./test_blast")
        blast2 = cb.blast.BLAST.load("./test_blast")
        hits2 = blast2.query(self.data)
        prediction2 = hits2.reconcile_models().filter(by="pval").annotate(["A1BG", "ZZZ3"]).fillna(0)
        self.assertTrue(np.all(prediction.values == prediction2.values))

    def test_align_blast(self):
        blast = cb.blast.BLAST(
            self.models, self.data, keep_exprs=True, n_posterior=20
        ).build_empirical()
        blast = blast.align(
            dict(query=self.data), path="tmp",
            epoch=3, patience=3
        ).build_empirical()
        hits = blast.query(self.data)
        prediction = hits.reconcile_models().filter(by="pval").annotate("cell_type1")
        blast.save("./test_blast")
        blast2 = cb.blast.BLAST.load("./test_blast", skip_exprs=True, mode="minimal")
        hits2 = blast2.query(self.data)
        prediction2 = hits2.reconcile_models().filter(by="pval").annotate("cell_type1")
        self.assertTrue(np.all(prediction.values == prediction2.values))


if __name__ == "__main__":
    # BLASTTest.setUpClass()
    # test = BLASTTest()
    # test.setUp()
    # test.test_posterior_dist_blast()
    # test.tearDown()
    unittest.main()
