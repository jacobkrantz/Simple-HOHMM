#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import sys

try:
    from setuptools.core import setup
except ImportError:
    from distutils.core import setup

with open('SimpleHOHMM/package_info.json') as f:
    _info = json.load(f)

def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(
        setup_requires=sphinx,
        name='SimpleHOHMM',
        version=_info["version"],
        author=_info["author"],
        author_email=_info["author_email"],
        packages=['SimpleHOHMM'],
        url='https://simple-hohmm.readthedocs.io',
        license='LICENSE.txt',
        description='High Order Hidden Markov Model for sequence classification',
        test_suite='test.test_suite',
    )

if __name__ == "__main__":
    setup_package()
