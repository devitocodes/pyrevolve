from setuptools import setup, Extension, find_packages
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
    from Cython.Compiler.Version import version as cython_version
    from packaging.version import Version
    ext = Extension("pyrevolve.crevolve", sources=["pyrevolve/schedulers/crevolve.pyx",
                                                   "src/revolve_c.cpp",
                                                   "src/revolve.cpp"],
                    include_dirs=[".", "pyrevolve"],
                    language="c++")

    compiler_directives = {}
    if Version(cython_version) >= Version("3.1.0"):
        compiler_directives["freethreading_compatible"] = True

    return cythonize([ext], compiler_directives=compiler_directives)


with open("README.md", "r") as fh:
    long_description = fh.read()

i_required = ["contexttimer", "numpy"]
s_required = ["cython>=3.0", "versioneer", "flake8"]

configuration = {
    'name': 'pyrevolve',
    'packages': find_packages(exclude=['examples', 'tests']),
    'setup_requires': s_required,
    'install_requires': i_required,
    'python_requires': '>=3.10,<3.14',
    'extras_require': {'compression': ['blosc2', 'pyzfp']},
    'ext_modules': lazy_cythonize(extensions),
    'version': versioneer.get_version(),
    'cmdclass': versioneer.get_cmdclass(),
    'description': "Python wrapper for Revolve checkpointing",
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'url': 'https://github.com/opesci/pyrevolve/',
    'author': "Imperial College London",
    'author_email': 'g.gorman@imperial.ac.uk',
    'license': 'MIT',
    'zip_safe': False,
    'classifiers': [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13'
    ]
}


setup(**configuration)
