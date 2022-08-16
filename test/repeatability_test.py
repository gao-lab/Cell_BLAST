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
cb.config.N_JOBS = 2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class DirectiRepeatabilityTest(unittest.TestCase):

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
        model2 = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, epoch=3, path="./test_directi"
        )
        latent2 = model2.inference(self.data)
        self.assertTrue(np.all(latent == latent2))

    def test_catgau(self):
        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, cat_dim=10, epoch=3, path="./test_directi"
        )
        latent = model.inference(self.data)
        model2 = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, cat_dim=10, epoch=3, path="./test_directi"
        )
        latent2 = model2.inference(self.data)
        self.assertTrue(np.all(latent == latent2))

    def test_semisupervised_catgau(self):
        '''
        self.data.obs.loc[
            cb.data.annotation_confidence(
                self.data,
                "cell_type1", return_group_percentile=True
            )[1] <= 0.5,
            "cell_type1"
        ] = ""
        '''
        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            supervision="cell_type1", latent_dim=10,
            epoch=3, path="./test_directi"
        )
        latent = model.inference(self.data)
        model2 = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            supervision="cell_type1", latent_dim=10,
            epoch=3, path="./test_directi"
        )
        latent2 = model2.inference(self.data)
        self.assertTrue(np.all(latent == latent2))

    def test_rmbatch(self):
        model = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            batch_effect="cell_type1", latent_dim=10,  # Just for test
            epoch=3, path="./test_directi"
        )
        latent = model.inference(self.data)
        model2 = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            batch_effect="cell_type1", latent_dim=10,  # Just for test
            epoch=3, path="./test_directi"
        )
        latent2 = model2.inference(self.data)
        self.assertTrue(np.all(latent == latent2))

    def test_blast(self):
        model1 = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, epoch=3, path="./test_directi"
        )
        blast1 = cb.blast.BLAST([model1], self.data)
        hits1 = blast1.query(self.data)#.filter("pval", 0.5)
        model2 = cb.directi.fit_DIRECTi(
            self.data, genes=self.data.uns["scmap_genes"],
            latent_dim=10, epoch=3, path="./test_directi"
        )
        blast2 = cb.blast.BLAST([model2], self.data)
        hits2 = blast2.query(self.data)#.filter("pval", 0.5)
        for h1, h2 in zip(hits1.pval, hits2.pval):
            self.assertTrue(np.all(h1 == h2))


if __name__ == "__main__":
    unittest.main()