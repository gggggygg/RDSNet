#!/usr/bin/env python

import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src", "RDSConv_ext")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    os.environ["CC"] = "g++"
    sources = main_file
    # extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]

        extra_compile_args["nvcc"] = [
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-std=c++17"
        ]


    else:
        raise NotImplementedError('Cuda is not available')
        pass

    print("Compiling sources:", sources)

    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "RDSConv_ext._C",  # ← 模块名必须含包路径
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="RDSConv",
    version="0.1",
    author="weipinhuan",
    url="",
    description="Radial-aware Deformable Sampling Convolution",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
