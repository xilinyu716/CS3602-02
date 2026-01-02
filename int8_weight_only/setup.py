from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='int8_weight_only',
    version='0.1.0',
    packages=find_packages(),  # Find python packages (int8_weight_only, etc if any)
    ext_modules=[
        CUDAExtension(
            name='int8_weight_only.w8a16_gemm', # Install as submodule of the python package
            sources=['int8_weight_only/binding.cpp', 'int8_weight_only/w8a16_gemm.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '--threads=4'
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=['torch']
)
