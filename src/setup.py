from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension("GBDT",
                sources=["GBDT.pyx", "../tree/ClassificationTree.cpp"],
                language="c++",
                include_dirs=['.', np.get_include()])

setup(name="GBDT",
      ext_modules=cythonize(ext))
