from setuptools import setup
import os
binscripts = [os.path.join("bin", f) for f in os.listdir("bin") if f.endswith(".py")]

setup(
    name="qcfitter",
    version="1.0",
    packages=['qcfitter'],
    package_dir={'': 'py/'},
    # package_data={"qsotools": ["tables/*"]},
    # include_package_data=True,
    scripts=binscripts,

    # install_requires=["docutils>=0.3"],

    # metadata to display on PyPI
    author = "Naim Goksel Karacayli",
    author_email = "ngokselk@gmail.com",
    description=("Quasar continuum fitter for DESI."),
    # url="https://bitbucket.org/naimgk/qsotools",

    # could also include long_description, download_url, etc.
)
    