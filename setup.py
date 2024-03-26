from setuptools import setup

with open("README.md","r") as fh:
    long_description = fh.read()
setup(
    name='frgd',
    version='0.0.1',
    description='Functional Renormalization Group Solver for Multiband Hamiltonians',
    packages=["frgd"],
    install_requires=['numba','numpy','python_version>="3.5"'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nkyirga/frg2d",
    author="Nahom Kifle Yirga",
    author_email="nkyirga@bu.edu"
)
