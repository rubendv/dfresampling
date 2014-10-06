from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext  =  [
    Extension("dfresampling_cython", sources=["dfresampling_cython.pyx"])
]

setup(
   name = "extensions", 
   cmdclass={'build_ext' : build_ext}, 
   include_dirs = [np.get_include()],   
   ext_modules=ext
   )
