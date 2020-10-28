import os
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

define_macros = []
extra_compile_args = {'cxx': []}
extra_link_args = []

define_macros += [('WITH_CUDA', None)]
nvcc_flags = os.getenv('NVCC_FLAGS', '')
nvcc_flags = [] if nvcc_flags == '' else nvcc_flags.split(' ')
nvcc_flags += ['-arch=sm_35', '--expt-relaxed-constexpr']
extra_compile_args['nvcc'] = nvcc_flags
extra_link_args += ['-lcusparse', '-l', 'cusparse']

setup(
    name='prebuilt_kernels',
    version='1.0.0',
    author='Simon Geisler',
    author_email='geisler@in.tum.de',
    description='Custom kernels for sparse topk and median',
    license='MIT',
    python_requires='>=3.6',
    ext_modules=[
        CUDAExtension(
            'prebuilt_kernels',
            [
                'csrc/custom.cpp',
                'csrc/custom_kernel.cu'
            ],
            #extra_cuda_cflags=['-lcusparse_static', '-lculibos']
            # extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            include_dirs=['/nfs/staff-ssd/geisler/anaconda3/envs/neurips20/lib/python3.7/site-packages/torch/lib']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
