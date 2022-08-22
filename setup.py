from setuptools import setup
from pathlib import Path

VERSION = "0.0.1"

with Path('requirements.txt').open() as f:
    INSTALL_REQUIRES = [line.strip() for line in f.readlines() if line]

setup(
    name = 'em_mlgssm',
    author = 'Ryohei Umatani',
    author_email = 'umataniryohei@gmail.com',
    url = 'https://github.com/ur17/em_mlgssm',
    description = 'Time series clustering with mlgssm',
    version = VERSION,
    packages = ['em_mlgssm'],
    install_requires = INSTALL_REQUIRES,
    test_suite = 'tests'
)