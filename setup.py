import os

from setuptools import setup

import torch

cuda_v = f"cu{torch.version.cuda.replace('.', '')}"
torch_v = torch.__version__.split('.')
torch_v = '.'.join(torch_v[:-1] + ['0'])

os.system(f'pip install torch-scatter==latest+{cuda_v} -f https://pytorch-geometric.com/whl/torch-{torch_v}.html')
os.system(f'pip install torch-sparse==latest+{cuda_v} -f https://pytorch-geometric.com/whl/torch-{torch_v}.html')
os.system(f'pip install torch-cluster==latest+{cuda_v} -f https://pytorch-geometric.com/whl/torch-{torch_v}.html')
os.system(f'pip install torch-spline-conv==latest+{cuda_v} -f https://pytorch-geometric.com/whl/torch-{torch_v}.html')

setup(
    name='rgnn',
    version='1.0.0',
    description='Reliable Graph Neural Networks via Robust Aggregation / Message Passing',
    author='Simon Geisler, Daniel Zügner, Stephan Günnemann',
    author_email='geisler@in.tum.de',
    packages=['rgnn'],
    install_requires=['torch-geometric'],
    zip_safe=False
)
