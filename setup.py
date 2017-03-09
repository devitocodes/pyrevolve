from distutils.core import setup, Extension
from Cython.Build import cythonize

ext_modules = [
  Extension("pyrevolve",
    sources=["python/pyrevolve.pyx",
             "c/revolve_c.cpp",
             "c/revolve.cpp"],
    language="c++",
  )
]

setup(
    ext_modules=cythonize(ext_modules),
)

