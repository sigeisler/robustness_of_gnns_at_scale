import logging
import subprocess

from setuptools import setup, find_packages

import torch

if torch.cuda.is_available():
    cuda_v = f"cu{torch.version.cuda.replace('.', '')}"
else:
    cuda_v = "cpu"

torch_v = torch.__version__.split('.')
torch_v = '.'.join(torch_v[:-1] + ['0'])


def system(command: str):
    output = subprocess.check_output(command, shell=True)
    logging.info(output)


system(f'pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
system(f'pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')

install_requires = [
    'filelock',
    'numba',
    'numpy',
    'pandas',
    'sacred',
    'scikit-learn',
    'scipy',
    'seaborn',
    'tabulate',
    'tinydb',
    'tinydb-serialization',
    'tqdm',
    'ogb',
    'torchtyping',
    'torch-geometric'
]

setup(
    name='rgnn_at_scale',
    version='1.0.0',
    author='Simon Geilser, Tobias Schmidt',
    description='Implementation & experiments for the paper "Robustness of Graph Neural Networks at Scale"',
    url='https://github.com/sigeisler/robustness_of_gnns_at_scale',
    packages=['rgnn_at_scale'] + find_packages(),
    install_requires=install_requires,
    zip_safe=False,
    package_data={'rgnn_at_scale': ['kernels/csrc/custom.cpp', 'kernels/csrc/custom_kernel.cu']},
    include_package_data=True
)
