import logging
import subprocess

from setuptools import setup

import torch

cuda_v = f"cu{torch.version.cuda.replace('.', '')}"
torch_v = torch.__version__.split('.')
torch_v = '.'.join(torch_v[:-1] + ['0'])


def system(command: str):
    output = subprocess.check_output(command, shell=True)
    logging.info(output)


system(f'pip install torch-scatter==latest+{cuda_v} -f https://pytorch-geometric.com/whl/torch-{torch_v}.html')
system(f'pip install torch-sparse==latest+{cuda_v} -f https://pytorch-geometric.com/whl/torch-{torch_v}.html')
system(f'pip install torch-cluster==latest+{cuda_v} -f https://pytorch-geometric.com/whl/torch-{torch_v}.html')
system(f'pip install torch-spline-conv==latest+{cuda_v} -f https://pytorch-geometric.com/whl/torch-{torch_v}.html')

install_requires = [
    'filelock',
    'numba',
    'pandas',
    'sacred',
    'scikit-learn',
    'scipy',
    'seaborn',
    'seml',
    'tabulate',
    'tinydb',
    'tinydb-serialization',
    'tqdm',
    'torch-geometric'
]

setup(
    name='rgnn_at_scale',
    version='1.0.0',
    description='Reliable Graph Neural Networks via Robust Aggregation / Message Passing',
    author='Simon Geisler, Daniel Zügner, Stephan Günnemann',
    author_email='geisler@in.tum.de',
    packages=['rgnn_at_scale'],
    install_requires=install_requires,
    zip_safe=False,
    package_data={'rgnn_at_scale': ['kernels/csrc/custom.cpp', 'kernels/csrc/custom_kernel.cu']},
    include_package_data=True
)
