.. _install:

Installation Guide
==================

We only support installation via pip right now.

Installation within virtual environments are recommended, see
https://virtualenv.pypa.io/en/latest/ or
https://conda.io/docs/user-guide/tasks/manage-environments.html.

For conda, here's a one-liner to set up an empty environment
for installing Cell BLAST:

``conda create -n cb python=3.6 && source activate cb``

Now follow the instructions below to install Cell BLAST:

1. Make sure you have a working version of tensorflow or tensorflow-gpu
   (version >= 1.5). You can follow the instructions on
   https://www.tensorflow.org/install/ about how to install tensorflow
   (and dependencies like CUDA and CuDNN for the GPU version), or just install
   via anaconda, which handles dependencies automatically:

   For installing the GPU supported version:
   ``conda install tensorflow-gpu=1.8``

   For installing the CPU only version:
   ``conda install tensorflow=1.8``

2. Install Cell BLAST by running:
   ``pip install Cell-BLAST``.

3. Check if the package can be imported in python interpreter:
   ``import Cell_BLAST as cb``

And you are good to go.

Feel free to contact us if you run into troubles during installation.
