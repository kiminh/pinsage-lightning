import setuptools

REQUIRED_PACKAGES = []

setuptools.setup(
    name="pinsage_lightning",
    version="0.1.0",
    description="Pytorch-lightning implementation of PinSAGE",
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
)
