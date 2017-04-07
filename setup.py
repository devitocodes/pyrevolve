from distutils.core import setup, Extension
from Cython.Build import cythonize

ext_modules = [
  Extension("pyrevolve.crevolve",
    sources=["pyrevolve/crevolve.pyx",
             "src/revolve_c.cpp",
             "src/revolve.cpp"],
    include_dirs = [".","pyrevolve"],
    language="c++",
  )
]

setup(
    name="pyrevolve",
    packages=["pyrevolve", "pyrevolve.crevolve"],
    ext_modules=cythonize(ext_modules)
)
