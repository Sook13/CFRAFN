from setuptools import setup, find_packages

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(
    name="torchvggish",
    version="0.2",
    author="Harri Taylor",
    author_email="taylorh23@cardiff.ac.uk",
    description="A Pytorch port of Tensorflow's VGGish embedding model.",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harritaylor/torchvggish",
    packages=find_packages(),
    license="Apache-2.0",
    install_requires=[
        "numpy",
        "torch",
        "resampy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.6',
)
