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

   entrypoints/analysis
   entrypoints/other
   entrypoints/plotting
   entrypoints/scripts


.. toctree::
   :caption: Modules
   :maxdepth: 1

   modules/definitions
   modules/harpy
   modules/kmod
   modules/model
   modules/optics_measurements
   modules/plotting
   modules/tbt
   modules/tune_analysis
   modules/utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
