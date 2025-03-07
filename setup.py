import setuptools
import os
from distutils.util import convert_path
from pathlib import Path

ver_path = convert_path('version.txt')
ver = None
if os.path.exists(ver_path):
    with open(ver_path) as ver_file:
        ver = ver_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="thermonet_dimensioning",
    version=ver,
    author='',
    author_email='',
    description="A dimensioning tool for Thermonets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[requirements],
    packages=setuptools.find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    python_requires='>=3.8.10',

)
