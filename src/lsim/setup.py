from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='slim_util',
    ext_modules=cythonize('slim_util.pyx'),
    include_dirs=[np.get_include()]
)

#  运行 python setup.py build_ext --inplace