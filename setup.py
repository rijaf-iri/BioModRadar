import os
from setuptools import setup, find_packages

def read(fname):
    try:
        with open(os.path.join(os.path.dirname(__file__), fname)) as fh:
            return fh.read()
    except IOError:
        return ''

requirements = read('requirements.txt').splitlines()
long_description = read('README.md')

setup(
    name='BioModRadar',
    version='1.0',

    author='Rija Faniriantsoa',
    author_email='rijaf@iri.columbia.edu',
    description='Radar data processing and biological echos model fitting.',
    url='https://github.com/rijaf-iri/BioModRadar',
    # long_description='Radar data processing and biological echos model fitting.',
    # long_description_content_type='text/plain',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    # packages=['BioModRadar'],

    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: National Meteorological and Hydrological Service',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
    install_requires=requirements,
)
