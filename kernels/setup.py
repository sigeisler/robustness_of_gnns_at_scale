import os

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    'cxx': [],
    'nvcc': []
}
CC = os.environ.get("CC", None)
if CC is not None:
    extra_compile_args["nvcc"].append("-ccbin={}".format(CC))
    extra_compile_args["nvcc"].extend(['-arch=sm_35', '--expt-relaxed-constexpr'])

setup(
    name='kernels',
    version='1.0.0',
    description='Custom kernels for sparse topk and median',
    license='MIT',
    python_requires='>=3.6',
    ext_modules=[
        CUDAExtension(
            'kernels',
            [
                'csrc/custom.cpp',
                'csrc/custom_kernel.cu'
            ],
            extra_link_args=['-lcusparse', '-l', 'cusparse'],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
