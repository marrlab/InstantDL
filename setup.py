import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "instantdl",
    version = "0.0.1",
    author = "Dominik Waibel & Ali Boushehri",
    author_email = "dominik.waibel@helmholtz-muenchen.de & ali.boushehri@roche.com",
    description = ("An easy and convenient Deep Learning pipeline for image segmentation and classification"),
    license = "MIT",
    keywords = "Computational Biology Deep Learning",
    url = "https://github.com/aliechoes/ICBPipeline",
    packages=['instantdl'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Image Recognition ",
        "License :: OSI Approved :: MIT License",
    ],
)