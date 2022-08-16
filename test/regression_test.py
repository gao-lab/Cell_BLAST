#!/usr/bin/env python

import sys
import os
import shutil
import unittest
import numpy as np
import anndata

sys.path.insert(0, "..")
import Cell_BLAST as cb
cb.config.RANDOM_SEED = 0

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DirectiTest(unittest.TestCase):

    def setUp(self):
        self.data = anndata.read_h5ad("pollen.h5ad")
        cb.data.normalize(self.data)

    def tearDown(self):
        if os.path.exists("./test_directi"):
            shutil.rmtree("./test_directi")

    def test_gau(self):
        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, epoch=3, path="./test_directi"
        )
        latent = model.inference(self.data)
        regression = cb.utils.read_hybrid_path("./regression_test/gau.h5//latent")
        deviation = np.abs(latent - regression).max()
        print(deviation)
        self.assertAlmostEqual(deviation, 0, places=6)

    def test_catgau(self):
        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, cat_dim=10, epoch=3, path="./test_directi"
        )
        latent = model.inference(self.data)
        regression = cb.utils.read_hybrid_path("./regression_test/catgau.h5//latent")
        deviation = np.abs(latent - regression).max()
        print(deviation)
        self.assertAlmostEqual(deviation, 0, places=6)

    def test_semisupervised_catgau(self):
        '''
        self.data.obs.loc[
            cb.data.annotation_confidence(self.data, "cell_type1")[1] <= 0.5,
            "cell_type1"
        ] = ""
        '''
        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, supervision="cell_type1",
            epoch=3, path="./test_directi"
        )
        latent = model.inference(self.data)
        regression = cb.utils.read_hybrid_path("./regression_test/semisupervised_catgau.h5//latent")
        deviation = np.abs(latent - regression).max()
        print(deviation)
        self.assertAlmostEqual(deviation, 0, places=6)

    def test_rmbatch(self):
        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, batch_effect="cell_type1",  # Just for test
            epoch=3, path="./test_directi"
        )
        latent = model.inference(self.data)
        regression = cb.utils.read_hybrid_path("./regression_test/rmbatch.h5//latent")
        deviation = np.abs(latent - regression).max()
        print(deviation)
        self.assertAlmostEqual(deviation, 0, places=6)


if __name__ == "__main__":
    unittest.main()