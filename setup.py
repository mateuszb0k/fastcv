from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fastcv",
    ext_modules=[
        CUDAExtension(
            name="fastcv",
            sources=[
                #"kernels/grayscale.cu",
                #"kernels/box_blur.cu",
                #"kernels/sobel.cu",
                #"kernels/dilation.cu",
                #"kernels/erosion.cu",
                "kernels/template_match.cu",
                "kernels/module.cpp",
            ],
            extra_compile_args={
                "cxx": [
                    "/O2",
                    "/std:c++17",
                    "/permissive-",
                    "/DNOMINMAX",
                    "/DWIN32_LEAN_AND_MEAN",
                ],
                "nvcc": [
                    "-O2",
                    "--std=c++17",
                    "-Xcompiler", "/permissive-",
                    "-Xcompiler", "/DNOMINMAX",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                    "--expt-extended-lambda",
                    "-DWIN32_LEAN_AND_MEAN",
            ]
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)

# To build the extension open x64 Native Tools Command Prompt for VS 2022
# Enter the directory where setup.py is located
# Paste these commands:
# set DISTUTILS_USE_SDK=1
# set CUDA_HOME= Path where you keep CUDA toolkit e.g. (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6)
# set USE_NINJA=0
# set SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
# And finnaly run:
# pip install -e . --no-build-isolation
