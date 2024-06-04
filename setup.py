from setuptools import find_packages, setup

name = "LstsqConvert"
version = "0.0.1"

setup(
    name=name,
    version=version,
    description="CoreML converter for torch.linalg.lstsq",
    url="https://github.com/dneprDroid/Lstsq-CoreML",
    author="dneprDroid",
    author_email="no@email.com",
    packages=['LstsqConvert'],
    package_dir={'LstsqConvert': 'converter'},
    install_requires=[
        "coremltools",
        "torch",
    ],
    python_requires=">=3.5.10",
)