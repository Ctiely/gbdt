from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension("GBDT",
                sources=["GBDT.pyx", "../tree/ClassificationTree.cpp"],
                language="c++",
                include_dirs=['.',
                              np.get_include(),
                              "/usr/local/opt/libomp/include"],
                extra_compile_args=["-L/usr/local/opt/libomp/lib", "-fopenmp", "-I/usr/local/Cellar/llvm/7.0.0/lib/clang/7.0.0/include"],
                extra_link_args=["-L/usr/local/opt/libomp/lib", "-fopenmp", "-I/usr/local/Cellar/llvm/7.0.0/lib/clang/7.0.0/include"])

setup(name="GBDT",
      ext_modules=cythonize(ext))