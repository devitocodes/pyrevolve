from setuptools import setup, Extension
import versioneer


class lazy_cythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None:
            self._list = self.callback()
        return self._list

    def __iter__(self):
        for e in self.c_list():
            yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


def extensions():
    from Cython.Build import cythonize
    ext = Extension("pyrevolve.crevolve", sources=["pyrevolve/crevolve.pyx",
                                                   "src/revolve_c.cpp",
                                                   "src/revolve.cpp"],
                    include_dirs=[".", "pyrevolve"], language="c++")
    return cythonize([ext])


configuration = {
    'name': 'pyrevolve',
    'packages': ["pyrevolve"],
    'setup_requires': ['cython>=0.17'],
    'ext_modules': lazy_cythonize(extensions),
    'version': versioneer.get_version(),
    'cmdclass': versioneer.get_cmdclass(),
    'description': "Python wrapper for Revolve checkpointing",
    'long_description': """pyrevolve is a python(ic) wrapper for
    the Revolve utility from the Griewank(2000) for optimal
    checkpointing. This is designed to work with existing
    code with minimal changes.""",
    'url': 'https://github.com/opesci/pyrevolve/',
    'author': "Imperial College London",
    'author_email': 'opesci@imperial.ac.uk',
    'license': 'MIT',
}


setup(**configuration)
