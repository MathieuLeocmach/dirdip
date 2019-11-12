import setuptools
# To use a consistent encoding
from codecs import open
from os import path



here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(
    name="texture", # Replace with your own username
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',
    # Choose your license
    license='GPL',
    author="Mathieu Leocmach",
    author_email="mathieu.leocmach@univ-lyon1.fr",
    description="Robust statistical tools to quantify discreet rearranging pattern. Implementation of Graner et al., Eur. Phys. J. E 25, 349-369 (2008) DOI 10.1140/epje/i2007-10298-8",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://cameleon.univ-lyon1.fr/mleocmach/texture",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy >= 0.18', 'numba', 'matplotlib', 'nose'],
)
