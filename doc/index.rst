.. Cell BLAST documentation master file, created by
   sphinx-quickstart on Tue Nov 27 01:35:22 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Cell BLAST's documentation!
======================================

**Cell_BLAST** is a single cell transcriptome querying tool, based on
a deep learning model, **DIRECTi**, which supports:

* Learning low dimensional cell embedding with intrinsic data clustering
* Semi-supervision
* Removal of batch effect / systematical bias

**Cell_BLAST** then performs query based on parametric cell embeddings from
**DIRECTi**, using posterior distribution distances.

Information like cell type annotation can then be transferred from reference
to query data based on Cell BLAST hits.


Contents
========

.. toctree::
   :maxdepth: 2

   install
   start
   doc


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
