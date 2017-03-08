from distutils.core import setup, Extension
from Cython.Build import cythonize

ext_modules = [
  Extension("pyrevolve",
    sources=["python/pyrevolve.pyx",
             "c/src.c"])
]

setup(
    ext_modules=cythonize(ext_modules),
)

