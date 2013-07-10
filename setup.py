from distutils.core import setup

setup(
    name='copper',
    version='0.0.4',
    author='Daniel Rodriguez',
    author_email='df.rodriguez143@gmail.com',
    packages=['copper', 'copper', 'copper.ml', 'copper.tests'],
    url='http://pypi.python.org/pypi/Copper/',
    license='LICENSE.txt',
    description='Fast, easy and intuitive machine learning prototyping.',
    long_description=open('README.txt').read(),
    install_requires=[
        "pandas>= 0.11",
        "scikit-learn == 0.13.1"
    ],
)
