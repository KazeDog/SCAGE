from distutils.core import setup
from pathlib import Path

from Cython.Build import cythonize
import numpy as np
from future.moves import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from _config import pdir

setup(
    ext_modules=cythonize(pdir + "/data_process/algos.pyx"),
    include_dirs=[np.get_include()]
)
