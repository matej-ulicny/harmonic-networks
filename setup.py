"""Setup for the harmonic package."""

import setuptools


with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Matej Ulicny",
    author_email="ulinm@tcd.ie",
    name='harmonic',
    license="BSD",
    description='Contains definitions of harmonic block implemented in pytorch that can be incorporated into neural network models.',
    version='0.1.0',
    long_description=README,
    url='https://github.com/matej-ulicny/harmonic-networks',
    packages=['harmonic'],
    install_requires=['numpy','torch'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)
