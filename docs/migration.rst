Migration Guide
===============

Version 0.5.0 sees a major change to the Cell BLAST code base. We switched the
implementation of the Cell BLAST model from Tensorflow to Pytorch, and substituted
our primary data format with the current standard :class:`anndata.AnnData`.

API migration
-------------

High-level APIs including :func:`Cell_BLAST.directi.fit_DIRECTi`, :func:`Cell_BLAST.directi.align_DIRECTi`
, :class:`Cell_BLAST.directi.DIRECTi`, :class:`Cell_BLAST.blast.BLAST` have been kept largely consistent.
The only small changes and cleanups are summarized in the following tables:

API changes of :func:`Cell_BLAST.directi.fit_DIRECTi`:

+---------------------------+-------------------------------------------------------------------------------------------+
|Arguments                  |Details of changes                                                                         |
+===========================+===========================================================================================+
|**optimizer**              |No longer supported, 'RMSpropOptimizer' used by default                                    |
+---------------------------+-------------------------------------------------------------------------------------------+

API changes of :func:`Cell_BLAST.directi.align_DIRECTi`:

+---------------------------+-------------------------------------------------------------------------------------------+
|Arguments                  |Details of changes                                                                         |
+===========================+===========================================================================================+
|**optimizer**              |No longer supported, 'RMSpropOptimizer' used by default                                    |
+---------------------------+-------------------------------------------------------------------------------------------+

For :class:`Cell_BLAST.blast.BLAST`, no API has changed.

Due to the backend switch, low-level APIs have been completely overhauled.

Data migration
--------------

Since the new version of Cell BLAST only support h5ad(anndata) format of data, we provide a function and a script to help to transform the original h5 format data to h5ad(anndata) format.

The function can be called as following:

.. code-block:: python
    :linenos:

    import Cell_BLAST as cb
    cb.data.h5_to_h5ad(inputfilename, outputfilename)

Or, you can run the `h5_to_h5ad.py <https://github.com/gao-lab/Cell_BLAST-dev/tree/master/DataMigration/h5_to_h5ad.py>`__ script we provided:

.. code-block:: bash
    :linenos:

    python h5_to_h5ad.py -i <inputfilename> -o <outputfilename>
