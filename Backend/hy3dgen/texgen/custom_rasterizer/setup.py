import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension, include_paths
import os
import subprocess
import sys

# Check if CUDA is available and nvcc is installed
def is_cuda_available():
    if not torch.cuda.is_available():
        return False
    try:
        subprocess.check_output(['nvcc', '--version'])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# build custom rasterizer
# build with `python setup.py install`

sources = [
    'lib/custom_rasterizer_kernel/rasterizer.cpp',
    'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
]

extra_compile_args = ['-std=c++17']
if sys.platform == 'darwin':
    extra_compile_args += ['-Wno-c++11-narrowing', '-stdlib=libc++']

define_macros = []
include_dirs = include_paths()

extra_link_args = []
if sys.platform == "darwin":
    # Add rpath for torch libs on macOS
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    extra_link_args.append(f"-Wl,-rpath,{torch_lib_path}")

if is_cuda_available():
    sources.append('lib/custom_rasterizer_kernel/rasterizer_gpu.cu')
    extension_cls = CUDAExtension
    define_macros.append(('WITH_CUDA', None))
    print("Building custom_rasterizer with CUDA support")
else:
    extension_cls = CppExtension
    print("Building custom_rasterizer with CPU support only")

custom_rasterizer_module = extension_cls(
    'custom_rasterizer_kernel',
    sources,
    include_dirs=include_dirs,
    define_macros=define_macros,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    packages=find_packages(),
    version='0.1',
    name='custom_rasterizer',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
