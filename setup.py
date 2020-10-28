from os import path

from setuptools import find_packages, setup

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements


PROJECT_ROOT = path.abspath(path.dirname(__file__))


def read_requirements(path):
    """Read a requirements file."""
    reqs = parse_requirements(path, session=False)

    return [str(ir.req) if hasattr(ir, "req") else str(ir.requirement) for ir in reqs]


setup(
    name="pinsage_lightning",
    version="0.1.0",
    description="Pytorch-lightning implementation of PinSAGE",
    install_requires=read_requirements(path.join(PROJECT_ROOT, "requirements.txt")),
    packages=find_packages(),
)
