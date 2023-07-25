# Robustness of Graph Neural Networks at Scale (NeurIPS 2021)

*Update: The attacks [GRBCD](https://pytorch-geometric.readthedocs.io/en/latest/modules/contrib.html#torch_geometric.contrib.nn.models.GRBCDAttack) and [PRBCD are now part of PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/contrib.html#torch_geometric.contrib.nn.models.PRBCDAttack)*

Here we provide the code and configuration for our NeurIPS 2021 paper "Robustness of Graph Neural Networks at Scale".

Other resources: [Project page](https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale/) - [Paper](https://arxiv.org/pdf/2110.14038.pdf) - [Video (Slideslive)](https://slideslive.com/38967603/robustness-of-graph-neural-networks-at-scale?ref=search)

Please cite our paper if you use the method in your own work:

```
@inproceedings{geisler2021_robustness_of_gnns_at_scale,
    title = {Robustness of Graph Neural Networks at Scale},
    author = {Geisler, Simon and Schmidt, Tobias and \c{S}irin, Hakan and Z\"ugner, Daniel and Bojchevski, Aleksandar and G\"unnemann, Stephan},
    booktitle={Neural Information Processing Systems, {NeurIPS}},
    year = {2021},
}
```

## Structure

Besides the standard python artifacts we provide:

- `cache`: for the pretrained models / attacked adjacency matrices
- `config`: the configuration files grouped by experiments
- `data`: for storing the datasets
- `experiments`: source code defining the types of experiments
- `kernels`: the custom kernel package
- `notebooks`: for (jupyter) notebooks
- `output`: for dumping the results of manual experiments (see instructions below)
- `rgnn_at_scale`: the source code
- `tests`: unit tests for some important parts of the code
- `script_execute_experiment.py`: the _main script_ to execute an experiment

## Installation

_Note: The setup is tested only for Linux 18.04 and will likely not work on other platforms._

For simplicity we recommend to install PyTorch with CUDA support a priori via anaconda:

```bash
conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

We used Python 3.7.6 and CUDA 10.2. We provide custom CUDA kernels that are fairly simple implementations for a `row-wise topk` and `row-wise weighted median` on a sparse matrix.

Due to custom CUDA kernels, you must be able to compile via `nvcc`. Conda handles the c++ compiler etc. You also must have installed the CUDA toolkit and should select the matching CUDA version for your environment. Note that PyTorch Geometric and PyTorch have some version-dependent restriction regarding the supported CUDA versions. See also [Build PyTorch from source](https://pytorch.org/get-started/locally/#mac-from-source) which captures the requirements for building custom extensions.

If you don't have access to a machine with a CUDA compatible GPU you can also use a CPU-only setup. However, note that the `soft-median` defense is only implemented using Custom CUDA kernels, hence not supported in a CPU-only setup.
Install pytorch for your CPU-only setup via anaconda:

```
conda install pytorch==1.8.1 torchvision torchaudio cpuonly -c pytorch
```

### Main Package

Thereafter we can install the actual module via (alternatively use `python install .`):

```bash
pip install -r requirements.txt
pip install .
```

By default the requirements are installed with very restrictive versioning since we did not test any other configuration. If you have version conflicts, you can also build without version restrictions via omitting the command `pip install -r requirements.txt` (not tested).

### Prebuilt Kernels [skipp this for CPU-only setup]

You also need to fulfill the [requirements for compiling a custom C++/CUDA extension for PyTorch](https://pytorch.org/tutorials/advanced/cpp_extension.html#using-your-extension) - usually satisfied by default via the conda command above.

You can either build the kernels a priori with

```bash
pip install ./kernels
```

or PyTorch will try to compile the kernels at runtime.

## Unit Tests

To (unit) test the robust mean functions, you can run (make sure pytest is on your path):

```bash
    pytest tests
```

We also provide the requirements we used during development via:

```bash
pip install -r requirements-dev.txt
```

## Minimum Working Example

As a minimum working example we provide a [Quick Start](notebooks/Quick_start_robustness_gnns_at_scale.ipynb) jupyter notebook, which can be run in colab. Here, we train a `Vanilla GCN` on the `Cora` dataset and attack it with local and global `PR-BCD`.

Further, the [Figure - Which nodes get attacked.ipynb](notebooks/Figure%20-%20Which%20nodes%20get%20attacked.ipynb) notebook shows the code used to analyze the learning curves and the distribution of attacked nodes (e.g. Fig. 2).

## Training

_Note: after open sourcing we will provide the full collection of pretrained models and in the case of transfer attacks we will also provide all perturbed adjacency matrices. For now we only include the pretrained models for Cora ML._

For the training and evaluation code we decided to provide Sacred experiments which make it very easy to run the same code from the command line or on your cluster.

To train or attack the models you can use the `script_execute_experiment` script and simply specif the respective configuration (if the configuration specifies `partition: gpu_large` you need at least 32 GB of GPU memory):

```bash
python script_execute_experiment.py --config-file 'config/train/cora_and_citeseer.yaml'
```

Alternatively, you can also execute the experiment directly passing the desired configuration:

```bash
python experiments/experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"Soft Median GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_median\", \"mean_kwargs\": {\"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.15, \"k\": 64}}" "artifact_dir=cache" "binary_attr=False"  "make_undirected=True"
```

By default all the results of the experiments will be logged into `./output`.

## Evaluation

For evaluation, we use the locally stored models in the `cache` folder (unless specified differently).

Similarly to training, we provide a script that runs the attacks for different seeds for all pretrained models. For all experiments, please check out the `config` folder. _Note: as this runs multiple seeds and budgets it will take several minutes to complete_

Additionally, we provide an example for a local attack on Cora ML and using PR-BCD (single seed and one budget):

```bash
python script_execute_experiment.py --config-files 'config/attack_evasion_local_direct/EXAMPLE_cora_and_citeseer_localprbcd.yaml'
```

## Perturbed Adjacency Matrices

We provide the perturbed adjacency matrices for a GCN as `torch_sparse.SparseTensor` for [`cora_ml`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/cora_ml.zip), [`citeseer`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/citeseer.zip) and [`pubmed`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/pubmed.zip).

Due to the storage requirements, we provide a list of added and removed edges for [`arxiv`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_arxiv.zip) and `products` (see table below). To restore the edge index see the following example where `pert_edge_index` is the edge index with applied perturbations:

```python
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
import torch


data = PygNodePropPredDataset(root='./datasets', name='ogbn-arxiv')[0]
edge_set = {(u.item(), v.item()) if u < v else (v.item(), u.item())
            for u, v in data.edge_index.T}

pert = np.load('./ogbn_arxiv_prbcd_budget_0p1_seed_1.npz')
pert_removed_set = {(u, v) for u, v in pert['pert_removed'].T}
pert_added_set = {(u, v) for u, v in pert['pert_added'].T}

pert_edge_set = edge_set - pert_removed_set | pert_added_set
pert_edge_index = torch.tensor(list(pert_edge_set)).T
```

| ↓ Attack / Budget → | 0.01 | 0.05 | 0.1 |
|---|---|---|---|
| GR-BCD | [`seed=0`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_greedyrbcd_budget_0p01_seed_0.npz) [`seed=1`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_greedyrbcd_budget_0p01_seed_1.npz) [`seed=5`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_greedyrbcd_budget_0p01_seed_5.npz) | [`seed=0`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_greedyrbcd_budget_0p05_seed_0.npz) [`seed=1`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_greedyrbcd_budget_0p05_seed_1.npz) [`seed=5`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_greedyrbcd_budget_0p05_seed_5.npz) | [`seed=0`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_greedyrbcd_budget_0p1_seed_0.npz) [`seed=1`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_greedyrbcd_budget_0p1_seed_1.npz) [`seed=5`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_greedyrbcd_budget_0p1_seed_5.npz) |
| PR-BCD | [`seed=0`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_prbcd_budget_0p01_seed_0.npz) [`seed=1`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_prbcd_budget_0p01_seed_1.npz) [`seed=5`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_prbcd_budget_0p01_seed_5.npz) | [`seed=0`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_prbcd_budget_0p05_seed_0.npz) [`seed=1`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_prbcd_budget_0p05_seed_1.npz) [`seed=5`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_prbcd_budget_0p05_seed_5.npz) | [`seed=0`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_prbcd_budget_0p1_seed_0.npz) [`seed=1`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_prbcd_budget_0p1_seed_1.npz) [`seed=5`](https://www.cs.cit.tum.de/fileadmin/w00cfj/daml/rgnns_at_scale/ogbn_products_prbcd_budget_0p1_seed_5.npz) |
