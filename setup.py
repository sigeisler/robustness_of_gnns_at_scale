import logging
import os
import subprocess
from warnings import warn

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
    'filelock==3.0.12',
    'numba==0.48.0',
    'pandas==1.0.1',
    'sacred==0.8.1',
    'scikit-learn==0.22.1',
    'scipy==1.4.1',
    'seaborn==0.10.0',
    'seml==0.2.0',
    'tinydb==4.2.0',
    'tinydb-serialization==2.0.0',
    'tqdm==4.42.1',
    'torch-geometric'
]
if os.getenv('RGNN_INSTALL_FLEXIBLE') is not None:
    logging.warning('No version restrictions of dependencies applied!')
    install_requires = [
        req.split('=')[0]
        for req
        in install_requires
    ]

setup(
    name='rgnn',
    version='1.0.0',
    description='Reliable Graph Neural Networks via Robust Aggregation / Message Passing',
    author='Simon Geisler, Daniel Zügner, Stephan Günnemann',
    author_email='geisler@in.tum.de',
    packages=['rgnn'],
    install_requires=install_requires,
    zip_safe=False
)
