from distutils.core import setup

setup(
    name='copper',
    version='0.0.1-alpha',
    author='Daniel Rodriguez',
    author_email='df.rodriguez143@gmail.com',
    packages=['copper', 'copper.core', 'copper.io', 'copper.explore', 'copper.test'],
    # scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    url='http://pypi.python.org/pypi/Copper/',
    license='LICENSE.txt',
    description='Tools for doing data analysis, exploration and machine learning in python using pandas and scikit-learn. Graphics in matplotlib and D3.js',
    long_description=open('README.txt').read(),
    install_requires=[
        "pandas>= 0.10",
        # "tornado == 2.4.1",
        # "scikit-learn == 0.14"
    ],
)
