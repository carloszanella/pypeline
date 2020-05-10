# coding=utf-8
import versioneer
from setuptools import setup, find_packages

packages = find_packages()

with open("requirements.txt") as fp:
    dependencies = fp.readlines()

# with open('requirements-test.txt') as fp:
#     test_dependencies = fp.readlines()

setup(
    name='trends-neuroimaging',
    description='Machine Learning Pipeline to Assess Brain Features',
    author='Data Revenue GmbH',
    author_email='markus@datarevenue.com',
    install_requires=dependencies,
    packages=packages,
    zip_safe=False,
    include_package_data=True,
)
