#!/usr/bin/env python
# -*- coding: utf8 -*

import setuptools

#Get the long description from the README file
with open("README.md", mode='r', encoding='utf-8') as fh:
    long_description = fh.read()

#Setup
setuptools.setup(
    name = 'ann',
    version = '0.0.1',
    author = 'Markus Pfeil',
    author_email = 'mpf@informatik.uni-kiel.de',
    description = 'Functions for the approximation of marine ecosystem model using artificial neural networks',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/todo', #TODO
    license='AGPL',
    packages = setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
