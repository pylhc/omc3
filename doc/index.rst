Welcome to omc3's documentation!
================================

``omc3`` is a library for beam optics measurements and corrections in particle accelerators used by the OMC team at `CERN <https://home.cern/>`_.

It consists of various methods for frequency analysis of circular particle accelerators turn-by-turn data, as well as beam optics properties computation and correction algorithms.
In addition, it consists of several easy-to-use entrypoint scripts for data analysis, results plotting and `MAD-X <https://mad.web.cern.ch/mad/>`_ wrapping.

Package Reference
=================

.. toctree::
   :caption: Main Entrypoints
   :maxdepth: 1
   :glob:

   entrypoints/*


.. toctree::
   :caption: Modules
   :maxdepth: 1
   :glob:

   modules/*

Citing
======

If ``omc3`` has been significant in your work, and you would like to acknowledge the package in your academic publication, please consider citing the following:

.. code-block:: bibtex

   @software{omc3,
   author    = {OMC-Team and Malina, L and Dilly, J and Hofer, M and Soubelet, F and Wegscheider, A and Coello De Portugal, J and Le Garrec, M and Persson, T and Keintzel, J and Garcia Morales, H and Tomás, R},
   doi       = {10.5281/zenodo.5705625},
   publisher = {Zenodo},
   title     = {OMC3},
   url       = {https://doi.org/10.5281/zenodo.5705625},
   year      = 2022
   }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
