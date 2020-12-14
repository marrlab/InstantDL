import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='instantdl',
    version='1.0.4',
    description='An easy and convenient Deep Learning pipeline for image segmentation and classification',
    author='Dominik Waibel, Ali Boushehri',
    author_email='dominik.waibel@helmholtz-muenchen.de, ali.boushehri@roche.com',
    license='MIT',
    keywords='Computational Biology Deep Learning',
    url='https://github.com/marrlab/InstantDL',
    packages=find_packages(exclude=['doc*', 'test*']),
    install_requires=[  'keras>=2.2.4',
                        'tensorboard>=1.13.0,<=1.15.2'],
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
    ],
)
