#!/usr/bin/env python3

import sys
import os
from distutils.core import setup

from setuptools import find_packages
from glob import glob


def add_package(package_data, package_name, extensions):        # TODO (keras) make sure if it is necessary in the plugin
    files = glob(package_name + "/**", recursive=True)
    files = list(map(lambda x: x[len(package_name)+1:].replace("\\", "/"), files))
    files = [f for f in files if any(f.endswith(e) for e in extensions)]
    package_data[package_name] = files


def find_package_data():
    package_data = {}

    extensions = [".cvlab", ".jpg", ".png", ".dcm", ".json", ".bmp", ".h5"]     # TODO (keras) what's the function of these extensions? should we add also txt?

    add_package(package_data, "../cvlab_keras", extensions)

    return package_data


if sys.version_info.major <= 2:
    raise Exception("Only python 3+ is supported!")


requirements = [
    "cvlab>=1.3.0rc1",
    "tensorflow>=2.3",       # TODO (keras) add required version at the end
]


__version__ = "0.1rc1"

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

description = long_description.splitlines()[0].strip()


setup(
    name="cvlab_keras",
    version=__version__,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Karolina Baryłka, Katarzyna Kaczorowska, Filip Chodziutko',
    url='https://github.com/cvlab-keras/cvlab_keras@test_installation',
    packages=find_packages(),
    package_data=find_package_data(),
    license="AGPL-3.0+",
    python_requires='>=3.5',
    install_requires=requirements,
)
