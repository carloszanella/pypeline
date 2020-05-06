#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import versioneer
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Data Revenue GmbH",
    author_email='alan@datarevenue.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    description="Trends NeuroImaging",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='manpy',
    name='manpy',
    packages=find_packages(include=['manpy']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/pedrocwb/manpy',
    version = versioneer.get_version(),
    cmdclass = versioneer.get_cmdclass(),
    zip_safe=False,
)