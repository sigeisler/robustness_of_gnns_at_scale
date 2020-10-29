from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='kernels',
    version='1.0.0',
    author='Simon Geisler',
    author_email='geisler@in.tum.de',
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
            extra_link_args=['-lcusparse', '-l', 'cusparse']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
