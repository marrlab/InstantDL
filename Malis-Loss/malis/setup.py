from distutils.core import setup
from Cython.Build import cythonize
import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("pairs_cython", ["pairs_cython.pyx", "malis_cpp.cpp"], language='c++',extra_link_args=["-std=c++11"],
                     extra_compile_args=["-std=c++11", "-w"])]

setup(cmdclass = {'build_ext': build_ext}, include_dirs=[numpy.get_include()], ext_modules = ext_modules)

# setup(
#     ext_modules=cythonize("pairs_cython.pyx"),
#     include_dirs=[numpy.get_include()]
# )  
