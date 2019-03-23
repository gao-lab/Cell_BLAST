.. _install:

Installation Guide
==================

We only support installation with local archive right now.

Installation within virtual environments are recommended, see
https://virtualenv.pypa.io/en/latest/ or
https://conda.io/docs/user-guide/tasks/manage-environments.html.

For conda, here's a one-liner to set up an empty environment
for ``Cell_BLAST``:

``conda create -n cb -c anaconda python=3.6 && source activate cb``

Now follow the instructions below to install ``Cell_BLAST``:

1. Make sure you have a working version of tensorflow or tensorflow-gpu
   (version >= 1.5). You can follow the instructions on
   https://www.tensorflow.org/install/ about how to install tensorflow
   (and dependencies like CUDA and CuDNN for the GPU version), or just install
   via the conda, which handles dependencies automatically:

   For installing the GPU supported version:
   ``conda install -c anaconda tensorflow-gpu=1.8``

   For installing the CPU only version:
   ``conda install -c anaconda tensorflow=1.8``

2. Download latest Cell_BLAST release from:
   https://github.com/gao-lab/Cell_BLAST/releases/.

3. Install by ``pip install Cell_BLAST-<version>.tar.gz``

And you are good to go.

Feel free to contact us if you run into troubles during installation.
