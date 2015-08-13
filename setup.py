from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext  =  [
    Extension("dfresampling.map_coordinates", sources=["dfresampling/map_coordinates.pyx"])
]

setup(
   name = "dfresampling", 
   version = "0.1",
   author = "Ruben De Visscher",
   cmdclass={'build_ext' : build_ext}, 
   include_dirs = [np.get_include()],   
   ext_modules=ext,
   packages=['dfresampling']
)
