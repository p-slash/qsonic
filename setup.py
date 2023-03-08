from setuptools import setup, find_namespace_packages
import os

binscripts = [os.path.join("bin", f) for f in os.listdir("bin")
              if not f.startswith(".")]
with open("requirements.txt") as file_reqs:
    requirements = file_reqs.read().splitlines()

setup(
    name="qcfitter",
    version="1.0",
    packages=find_namespace_packages(where='py'),
    package_dir={'': 'py/'},
    scripts=binscripts,
    install_requires=requirements,

    # metadata to display on PyPI
    author="Naim Goksel Karacayli",
    author_email="ngokselk@gmail.com",
    description=("Quasar continuum fitter for DESI."),
    # could also include long_description, download_url, etc.
)
