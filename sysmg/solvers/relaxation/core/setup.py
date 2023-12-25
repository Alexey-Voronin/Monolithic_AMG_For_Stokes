from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys


# Include directories for pybind11 and numpy
try:
    import pybind11

    pybind11_includes = pybind11.get_include()
except ImportError:
    pybind11_includes = ""

try:
    import numpy

    numpy_includes = numpy.get_include()
except ImportError:
    numpy_includes = ""


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        "msvc": ["/EHsc"],
        "unix": [],
    }
    l_opts = {
        "msvc": [],
        "unix": [],
    }

    if sys.platform == "darwin":
        darwin_opts = ["-stdlib=libc++", "-mmacosx-version-min=10.14"]
        c_opts["unix"] += darwin_opts
        l_opts["unix"] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == "unix":
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append("-std=c++11")
            link_opts.append("-std=c++11")
        elif ct == "msvc":
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args += opts
            ext.extra_link_args += link_opts
        super(BuildExt, self).build_extensions()


if sys.platform == "linux":
    openblas_include = "/home/voronin2/OpenBLAS/build/include/"
    openblas_lib = "/home/voronin2/OpenBLAS/build/lib/"
    arch_compile_flags = ["-ftree-vectorize"]
elif sys.platform == "darwin":  # macOS
    openblas_include = "/opt/homebrew/opt/openblas/include/"
    openblas_lib = "/opt/homebrew/opt/openblas/lib/"
    arch_compile_flags = ["-fvectorize"]
else:
    raise Exception(f"Unsupported platform: {sys.platform}")

ext_modules = [
    Extension(
        "patch_mult",
        ["patch_mult.cpp"],
        include_dirs=[pybind11_includes, numpy_includes, openblas_include],
        language="c++",
        libraries=["openblas"],
        library_dirs=[openblas_lib],
        extra_compile_args=[
            "-Wall",
            "-Wextra",
            "-std=c++11",
            "-O3",
            "-march=native",
            "-funroll-loops",
            "-ffast-math",
        ]
        + arch_compile_flags,
        extra_link_args=["-L" + openblas_lib, "-lopenblas", "-flto"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
