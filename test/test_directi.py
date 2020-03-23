#!/usr/bin/env python

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class DirectiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = cb.data.ExprDataSet.read_dataset(
            "pollen.h5"
        ).normalize()

    def tearDown(self):
        if os.path.exists("./test_directi"):
            shutil.rmtree("./test_directi")

    def test_gau(self):
        model = cb.directi.fit_DIRECTi(
            self.data, latent_dim=10, epoch=3, path="./test_directi"
        )
        self.data.latent = model.inference(self.data)
        self.assertFalse(np.any(np.isnan(self.data.latent)))
        with self.assertRaises(Exception):
            model.clustering(self.data)
        _ = self.data.visualize_latent("cell_type1", method="tSNE", sort=True, dr_kws=dict(n_iter=250))
        _ = self.data.visualize_latent("cell_type1", method="tSNE", reuse=False, sort=True, dr_kws=dict(n_iter=250))
        _ = self.data.visualize_latent("cell_type1", method="UMAP", random_seed=123, dr_kws=dict(n_epochs=20))
        _ = self.data.visualize_latent("cell_type1", method="UMAP", random_seed=123, dr_kws=dict(n_epochs=20))
        _ = self.data.visualize_latent("cell_type1", method=None)
        with self.assertRaises(ValueError):
            _ = self.data.visualize_latent("cell_type1", method="NA")
        model.save()
        model.close()
        del model
        model = cb.directi.DIRECTi.load("./test_directi")
        latent2 = model.inference(self.data)
        self.assertTrue(np.all(self.data.latent == latent2))

    def test_catgau(self):
        with self.assertRaises(ValueError):
            _ = self.data.visualize_latent("cell_type1", method="tSNE", dr_kws=dict(n_iter=250))
        model = cb.directi.DIRECTi(
            genes=self.data.uns["scmap_genes"],
            latent_module=cb.latent.CatGau(
                latent_dim=10, cat_dim=20, multiclass_adversarial=True,
                cat_merge=True, min_silhouette=0.2, patience=2
            ),
            rmbatch_modules=[],
            prob_module=cb.prob.MSE(),
            path="./test_directi"
        ).compile("RMSPropOptimizer", 1e-3)
        model.fit(
            cb.utils.DataDict(
                exprs=self.data[:, self.data.uns["scmap_genes"]].exprs,
                library_size=np.array(self.data.exprs.sum(axis=1)).reshape((-1, 1))
            ), epoch=300, patience=20
        )
        _ = model.clustering(self.data)
        self.data.latent = model.inference(self.data)
        self.assertFalse(np.any(np.isnan(self.data.latent)))
        model.save()
        model.close()
        del model
        model = cb.directi.DIRECTi.load("./test_directi")
        latent2 = model.inference(self.data)
        self.assertTrue(np.all(self.data.latent == latent2))
        random_state = np.random.RandomState(0)
        model.gene_grad(
            self.data, random_state.randn(self.data.shape[0], 10)
        )

    def test_semisupervised_catgau(self):
        _ = self.data.annotation_confidence(
            "cell_type1", used_vars="scmap_genes",
            return_group_percentile=False
        )
        self.data.obs.loc[
            self.data.annotation_confidence(
                "cell_type1", return_group_percentile=True
            )[1] <= 0.5, "cell_type1"
        ] = ""
        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, cat_dim=20, prob_module="ZINB",
            supervision="cell_type1", epoch=3, path="./test_directi"
        )
        self.data.latent = model.inference(self.data)
        self.assertFalse(np.any(np.isnan(self.data.latent)))
        model.save()
        model.close()
        del model
        model = cb.directi.DIRECTi.load("./test_directi")
        latent2 = model.inference(self.data)
        self.assertTrue(np.all(self.data.latent == latent2))

    def test_rmbatch(self):
        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, batch_effect="cell_type1",  # Just for test
            epoch=3, path="./test_directi"
        )
        self.data.latent = model.inference(self.data)
        self.assertFalse(np.any(np.isnan(self.data.latent)))
        model.save()
        model.close()
        del model
        model = cb.directi.DIRECTi.load("./test_directi")
        latent2 = model.inference(self.data)
        self.assertTrue(np.all(self.data.latent == latent2))

        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            prob_module="NB", latent_dim=10, depth=0,
            batch_effect="cell_type1",  # Just for test
            rmbatch_module="RMBatch",
            epoch=3, path="./test_directi"
        )
        self.data.latent = model.inference(self.data)
        self.assertFalse(np.any(np.isnan(self.data.latent)))
        model.save()
        model.close()
        del model
        model = cb.directi.DIRECTi.load("./test_directi")
        latent2 = model.inference(self.data)
        self.assertTrue(np.all(self.data.latent == latent2))

        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            prob_module="ZINB",
            latent_dim=10, batch_effect="cell_type1",  # Just for test
            rmbatch_module="MNN",
            epoch=3, path="./test_directi"
        )
        self.data.latent = model.inference(self.data)
        self.assertFalse(np.any(np.isnan(self.data.latent)))
        model.save()
        model.close()
        del model
        model = cb.directi.DIRECTi.load("./test_directi")
        latent2 = model.inference(self.data)
        self.assertTrue(np.all(self.data.latent == latent2))

        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            prob_module="ZILN",
            latent_dim=10, batch_effect="cell_type1",  # Just for test
            rmbatch_module="MNNAdversarial",
            epoch=3, path="./test_directi"
        )
        self.data.latent = model.inference(self.data)
        self.assertFalse(np.any(np.isnan(self.data.latent)))
        model.save()
        model.close()
        del model
        model = cb.directi.DIRECTi.load("./test_directi")
        latent2 = model.inference(self.data)
        self.assertTrue(np.all(self.data.latent == latent2))

        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            prob_module="LN",
            latent_dim=10, batch_effect="cell_type1",  # Just for test
            rmbatch_module="AdaptiveMNNAdversarial",
            epoch=3, path="./test_directi"
        )
        self.data.latent = model.inference(self.data)
        self.assertFalse(np.any(np.isnan(self.data.latent)))
        model.save()
        model.close()
        del model
        model = cb.directi.DIRECTi.load("./test_directi")
        latent2 = model.inference(self.data)
        self.assertTrue(np.all(self.data.latent == latent2))


if __name__ == "__main__":
    # DirectiTest.setUpClass()
    # test = DirectiTest()
    # test.test_catgau()
    # test.tearDown()
    unittest.main()
