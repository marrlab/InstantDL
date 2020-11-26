from Cython.Distutils import build_ext
import setuptools
import numpy as np

ext_modules = [setuptools.extension.Extension("malis.pairs_cython",
               ["malis/pairs_cython.pyx","malis/malis_cpp.cpp"],
               language='c++',
               extra_compile_args=["-std=c++14"],
               extra_link_args=["-std=c++14"],
               include_dirs=[np.get_include()])]

setuptools.setup(name="malis",
                 version="1.0",
                 cmdclass={'build_ext': build_ext},
                 ext_modules=ext_modules,
                 install_requires=['cython','numpy','h5py','scipy'],
	         setup_requires=['cython','numpy','scipy'],
                 packages=["malis"])
