from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fourier_layer',
    ext_modules=[
        CUDAExtension('fourier_layer_cuda', [
            'fourier_layer_cuda.cpp',
            'fourier_layer_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })