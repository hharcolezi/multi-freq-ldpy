import setuptools
from setuptools import setup, find_packages
import os

with open("README.md", "r") as fh:
    describe_package = fh.read()

setup(
    name='multi-freq-ldpy',
    version='0.2.3',
    license='MIT',
    author="Héber H. Arcolezi",
    author_email='hh.arcolezi@gmail.com',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    url='https://github.com/hharcolezi/multi-freq-ldpy',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='local-differential-privacy',
    install_requires=[
                        'numpy', 'numba', 'xxhash'
                     ],
    description='Multiple Frequency Estimation Under Local Differential Privacy in Python',
    long_description=describe_package,
    python_requires=">=3.6",

)