from setuptools import find_packages, setup

install_requirements = [
    "numpy == 1.20.0",
    "matplotlib == 3.4.1",
    "xgboost == 1.3.3",
    "scikit-learn == 0.24.1",
    "pandas == 1.2.3",
    "category-encoders == 2.2.2",
    "compress_pickle[lz4] == 2.0.1",
    "py4j == 0.10.9.2",
    "coolname == 1.1.0",
    "pymongo == 3.11.3",
    "geopython == 1.0.0",
    "geopandas == 0.9.0",
    "turfpy == 0.0.6",
    "tqdm == 4.60.0",
    "rtree == 0.9.7",
    "python-igraph == 0.9.1",
    "networkx == 2.5.1",
    "h3 == 3.7.2",
    "folium == 0.12.1",
    "xlrd == 2.0.1",
    "openpyxl == 3.0.7",
    "comet-ml == 3.9.0",
    "jupyter == 1.0.0",
    "seaborn == 0.11.1",
]

dev_requirements = ["black", "isort", "ipython"]

setup(
    name="savvy-freight-eta",
    version="1.0.0",
    url="https://github.com/tryolabs/cfl-savvy-freight-eta",
    author="Diego Kiedanski",
    author_email="dkiedanski@tryolabs.com",
    description="Predicting the ETA of Freight Trains using Savvy sensors",
    packages=find_packages(),
    install_requires=install_requirements,
    extras_require={
        "dev": dev_requirements,
    },
)
