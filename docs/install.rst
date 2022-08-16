.. _install:

Installation Guide
==================

We only support installation via pip right now.

Installation within virtual environments are recommended, see
https://virtualenv.pypa.io/en/latest/ or
https://conda.io/docs/user-guide/tasks/manage-environments.html.

For conda, here's a one-liner to set up an empty environment
for installing Cell BLAST:

.. code-block:: bash
   :linenos:

   conda create -n cb python=3.9 && source activate cb

Then follow the instructions below to install Cell BLAST:

1. Install Cell BLAST by running:

   .. code-block:: bash
      :linenos:

      pip install Cell-BLAST

2. Check if the package can be imported in Python interpreter:

   .. code-block:: python
      :linenos:

      import Cell_BLAST as cb

And you are good to go.

Feel free to contact us if you run into troubles during installation.
