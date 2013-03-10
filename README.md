Copper
======

Wrapper around python data analysis packages such as pandas, scikit-learn and matplotlib
to make data analysis on python easier.

Requirements
------------

1. Python
2. **pandas**
3. scikit-learn
4. matplotlib
5. rpy2
6. tornado

Note: pandas is the only package that is required before installing copper, but is
recommended to have all other packages installed too.

Note 2: The package is developed for Python 3 and Python 2 with a single code base. But the main target is Python 3 so is recommended since most packages already support Python 3.

Install
-------

`pip install copper`

Features
--------

1. Project structure for Data Analysis projects ala [Project Template](http://www.johnmyleswhite.com/notebook/2010/08/26/projecttemplate/) on R.
2. Dataset: Wrapper around pandas.DataFrame to introduce metadata
3. Data transformation templates
4. Custom matplotlib charts for exploration: histograms, scatterplots
5. Exploration via D3.js (very experimental)
6. More data imputation options via R (rpy2)
7. Rapid Machine Learning prototyping:
    * Easy to compare classifiers
    * Ensemble (bagging)

Project Structure
-----------------

Copper uses a project structure based on [Project Template](http://www.johnmyleswhite.com/notebook/2010/08/26/projecttemplate/) (from R) to give structure to a Data Analysis project.

The suggested structure is:

`data` -> `project/data': All the data files, raw, cached, etc.

Is suggested to use `/data/raw` for raw files such as `.csv` files.

Copper by default loads data from the `data` folder. For example: `data = copper.read_csv('catalog.csv')` will load the `project/data/catalog.csv` file into a pandas.DataFrame using the pandas `read_csv` method and parameters.

As expected when saving files (`copper.save(...)` or `copper.export(...)`) copper saves the files on the `data` folder

`source` -> `src`: Python, iPython notebook files.

Following the intuition every file inside the `source` folder should do:

```python
import copper
copper.project.path = '../'
```

For other suggested folders see: [Project Template](http://www.johnmyleswhite.com/notebook/2010/08/26/projecttemplate/)

For more info about this see the examples below.

Examples
--------

* [Project structure and histograms](http://nbviewer.ipython.org/urls/raw.github.com/danielfrg/copper/master/examples/donors/src/explore.ipynb)

Catalog:
* [Custom transformation and basic machine learning](http://nbviewer.ipython.org/urls/raw.github.com/danielfrg/copper/master/examples/catalog/src/ml.ipynb)

Loans:
* [Automatic data transformation](http://nbviewer.ipython.org/urls/raw.github.com/danielfrg/copper/master/examples/loans/src/transform.ipynb)

Kaggle Bulldozers:
* Post: [Cleaning - Datasets metadata and join Datasets](http://danielfrg.github.com/blog/2013/03/07/kaggle-bulldozers-basic-cleaning/)

For more information and more examples (but some are possible outdated) can see my blog: [danielfrg.github.com](http://danielfrg.github.com)
