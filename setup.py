import os
import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="BBB",
    version="0.0.1",
    author="Max Bronckers, Yufan Wang, Alan Clark",
    author_email="ajc348@cam.ac.uk",
    description="BBB implementation for MLMI4",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cambridge-Uni-MLMI-Advanced-ML-21-22-G7/BBB",
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "bbb"},
    packages=setuptools.find_packages(where="bbb"),
    python_requires=">=3.6",
    install_requires=required,
)