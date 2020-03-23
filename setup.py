import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Cell_BLAST",
    version="0.3.6",
    author="Zhijie Cao",
    author_email="caozj@mail.cbi.pku.edu.cn",
    description="Single-cell transcriptome querying tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gao-lab/Cell_BLAST",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "numba>=0.39.0",  # minimal version supporting np.unique
        "scipy>=1.0.0",  # minimal version supporting scipy.stats.energy_distance
        "joblib>=0.12.0",  # minimal version supporting locky backend
        "scikit-learn>=0.17.0",  # normalization issue
        "tqdm>=4.12.0",  # deprecation warning in the sys module
        "pandas>=0.21.0",  # optional, anndata requirement
        "h5py>=2.7.0",  # optional, anndata requirement
        "python-igraph>=0.7.1",  # only version on conda at the time of packaging
        "pronto>=0.10.2",  # minimal version supporting python 3.6
        "seaborn>=0.9.0", # minimal version supporting sns.scatterplot
        "umap-learn>=0.2.1",  # scipy.sparse.csgraph issue
        "anndata>=0.6.14",  # minimal version supporting write_h5ad
        "loompy>=2.0.6",  # minimal version to pass tests
        "statsmodels>=0.8.0",  # python and numpy version compatibility
        "plotly"
    ]
)
