.. copper documentation master file, created by
   sphinx-quickstart on Sun Jul  7 21:29:06 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
============

Copper objective is to make machine learning prototyping on python as fast and intuitive as possible.
To do so connects via the Dataset and Model Comparison classes the pandas and
scikit-learn projects.

Source is on Github_.

.. image:: https://api.travis-ci.org/danielfrg/copper.png

Examples
--------

1. `Iris classification`_

Requirements
------------

1. pandas
2. scikit-learn

Note: The package is currently developed for Python 2.7 because scikit-learn does not support
python 3 yet. When scikit-learn supports python 3 this project will drop support for python 2 and
support only python 3.

I recommend using the Anaconda_ python distribution.

Install
-------

``pip install copper``

.. _Iris classification: http://nbviewer.ipython.org/urls/raw.github.com/danielfrg/copper/master/docs/examples/iris/iris.ipynb
.. _Anaconda: http://docs.continuum.io/anaconda/index.html
.. _Github: https://github.com/danielfrg/copper

.. toctree::
    :maxdepth: 3

    genindex
    :ref:`modindex`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

