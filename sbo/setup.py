#!/usr/bin/env python
# -*- coding: utf8 -*

import setuptools

#Get the long description from the README file
with open("README.md", mode='r', encoding='utf-8') as fh:
    long_description = fh.read()

#Setup
setuptools.setup(
    name = 'sbo',
    version = '0.0.1',
    author = 'Markus Pfeil',
    author_email = 'mpf@informatik.uni-kiel.de',
    description = 'Functions for the surrogate based optimization of a marine ecosystem model',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/slawig/bgc-ann/tree/master/sbo',
    license='AGPL',
    packages = setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

