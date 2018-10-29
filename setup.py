from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="mpCMF",
    version="0.1",
    author="Pedro F. Ferreira",
    description="Package for modified pCMF.",
    packages=find_packages(),
    install_requires=requirements,
)
