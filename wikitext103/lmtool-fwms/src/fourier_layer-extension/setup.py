from setuptools import setup, Extension
#from distutils.core import setup, Extension
from torch.utils import cpp_extension

setup(name='fourier_layer_cpp',
      ext_modules=[cpp_extension.CppExtension('fourier_layer_cpp', ['fourier_layer.cpp'] , extra_compile_args = ['-fopenmp', '-fpic'], extra_link_args = ['-lgomp'] ) ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
# #,  extra_compile_args = ["-fopenmp"], extra_link_args = ["-fopenmp"]


# ext = Extension(
   # name='fourier_layer_cpp',
   # sources=['fourier_layer.cpp'],
   # include_dirs=cpp_extension.include_paths(),
   # extra_compile_args=['-fopenmp'],
   # extra_link_args=['-lgomp'],
   # language='c++')




