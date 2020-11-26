from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os

source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waterz')
include_dirs = [
    source_dir,
    os.path.join(source_dir, 'backend'),
    # os.path.dirname(get_python_inc()),
    numpy.get_include(),
]
extensions = [
    Extension(
        'waterz.evaluate',
        sources=['waterz/evaluate.pyx', 'waterz/frontend_evaluate.cpp'],
        include_dirs=include_dirs,
        language='c++',
        extra_link_args=['-std=c++11'],
        extra_compile_args=['-std=c++11', '-w'])
]

setup(
        name='waterz',
        version='0.8',
        description='Simple watershed and agglomeration for affinity graphs.',
        url='https://github.com/funkey/waterz',
        author='Jan Funke',
        author_email='jfunke@iri.upc.edu',
        license='MIT',
        requires=['cython','numpy'],
        packages=['waterz'],
        package_data={
            '': [
                'waterz/*.h',
                'waterz/*.hpp',
                'waterz/*.cpp',
                'waterz/*.cpp',
                'waterz/*.pyx',
                'waterz/backend/*.hpp',
            ]
        },
        include_package_data=True,
        zip_safe=False,
        ext_modules=cythonize(extensions)
)
