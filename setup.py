"""This file is for setup information and version control used."""
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme_text = f.read()

setup(
    project_name='service-ncku-derms',
    project_version='0.0.1',
    description='In this project, we will develop DERMS application for EMS system',
    long_description=readme_text,
    author='Hank Liang',
    url='',
    packages=find_packages(exclude=('tests'))
)
