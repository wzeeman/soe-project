from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

module =    [ Extension(    'preProcess', ["preProcess.pyx"],
                            include_dirs = [ numpy.get_include() ],
                            #extra_compile_args=['-fopenmp'],
                            #extra_link_args=['-fopenmp']
                        )

            ]

setup( name='preProcess', ext_modules=cythonize(module))

# modules = [
#     Extension(
#         "hello",
#         ["parallel.pyx"],
#         extra_compile_args=['-fopenmp'],
#         extra_link_args=['-fopenmp'],
#     )
# ]
