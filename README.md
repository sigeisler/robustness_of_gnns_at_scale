# Reliable Graph Neural Networks via Robust Location Estimation

This repository contains the official implementation of `Reliable Graph Neural Networks via Robust Location Estimation`.

**TODO: Link Arxiv, Video, Project Page**

The main idea is to substitute the message passing aggregation in a Graph Neural Network (GNN)

![Aggregation](https://latex.codecogs.com/gif.download?%5Cmathbf%7Bh%7D%5E%7B%28l%29%7D_v%20%3D%20%5Csigma%5E%7B%28l%29%7D%20%5Cleft%28%20%5Ctext%7BAGGREGATE%7D%5E%7B%28l%29%7D%20%5Cleft%20%5C%7B%20%5Cleft%28%20%5Cmathbf%7BA%7D_%7Bvu%7D%2C%20%5Cmathbf%7Bh%7D%5E%7B%28l-1%29%7D_u%20%5Cmathbf%7BW%7D%5E%7B%28l%29%7D%20%5Cright%29%2C%20%5Cforall%20%5C%2C%20u%20%5Cin%20%5Cmathcal%7BN%7D%28v%29%20%5Ccup%20v%20%5Cright%20%5C%7D%20%5Cright%29)

with robust location estimators for improved robustness w.r.t. adversarial modifications of the graph structure.

In Figure 1 of our paper, we give an exemplary plot for Nettack that clearly shows that strong adversarially added edges are resulting in a concentrated region of outliers. This is exactly the case where robust aggregations are particularity strong.
![Figure 1](./assets/aggregation.png)

We show that in combination with personalized page rank (aka GDC - Graph Diffusion Convolution) our method outperforms all baselines and tested state of the art adversarial defenses:
![Figure 5](./assets/cert_rob_over_degree.png)


## TL;DR
Execute
```bash
conda install pytorch==1.6.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
pip install .

cd prebuilt_kernels
pip install .
cd ../sparse_smoothing
conda install "gmpy2==2.1.0b1" "statsmodels==0.12"
pip install .
cd ..
```
for setting the project up. Run for the results on empirical robustness (takes about 4 minutes with a GPU):
```bash
python script_evaluate_empirical_robustness.py
```
For the certified robustness via randomized smoothing use:
```bash
python script_evaluate_certified_robustness.py
```

## Requirements

For simplicity we recommend to install PyTorch a priori via anaconda:
```bash
conda install pytorch==1.6.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
```
We used Python 3.7.6 and CUDA 10.1. Note that PyTorch Geometric and PyTorch have some version-dependent restriction regarding the supported CUDA versions. Due to custom CUDA kernels, it can become tricky since your local environment needs to comply with the PyTorch wheel e.g. obtained via `pip`. The custom CUDA kernels provide fairly simple implementations for a `row-wise topk` and `row-wise weighted median` on a sparse matrix.

Thereafter we can install the actual module via (alternatively use `python install .`):
```bash
pip install .
```
By default the requirements are installed with very restrictive versioning since we did not test any other configuration. If you have version conflicts, you can also build without version restrictions via `RGNN_INSTALL_FLEXIBLE=1 pip install .` (not tested).

### Prebuilt Kernels

In case you want to use the GPU, you also need to fulfill the [requirements for compiling a custom C++/CUDA extension for PyTorch](https://pytorch.org/tutorials/advanced/cpp_extension.html#using-your-extension) - usually satisfied by default voa the conda command above.

You can either build the kernels a priori with
```bash
cd prebuilt_kernels
pip install .
cd ..
```
or PyTorch will try to compile the kernels at runtime.

### Sparse Smoothing

If you want to run the randomized smoothing experiments you need to install the respective module:
```bash
cd sparse_smoothing
conda install "gmpy2==2.1.0b1" "statsmodels==0.12"
pip install .
cd ..
```

In case the installation of `gmpy` fails please check out their [installation guide](https://gmpy2.readthedocs.io/en/latest/intro.html#installation).

## Unit Tests

To (unit) test the robust mean functions, you can run (make sure pytest is on your path):

```bash
    pytest tests
```

## Training

**Note: you can skip this section as we provide pretrained models**

For the training and evaluation code we decided to provide SEML/Sacred experiments which make it very easy to run the same code from the command line or on your cluster.

The training for all the pretrained models is bundled in:
```bash
python script_train.py --kwargs '{"artifact_dir": "cache"}'
```

To train a model on `cora_ml` for evaluating the empirical robustness (from then on it will be also used for evaluation) e.g. run:
```bash
python experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.15, \"k\": 64}}" "artifact_dir=cache" "binary_attr=False"
```
With binary attributes (for randomized smoothing) use:
```bash
python experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.15, \"k\": 64}}" "artifact_dir=cache" "binary_attr=True"
```

## Evaluation

For evaluation, we execute all locally stored (pretrained) models.

### Empirical Robustness

Similarly to training, we provide a script that runs the attacks for different seeds for all pretrained models:
```bash
python script_evaluate_empirical_robustness.py
```

This will print the following table:

|                                         | ('fgsm', 0.0)   | ('fgsm', 0.1)   | ('fgsm', 0.25)   | ('pgd', 0.0)   | ('pgd', 0.1)   | ('pgd', 0.25)   |
|:----------------------------------------|:----------------|:----------------|:-----------------|:---------------|:---------------|:----------------|
| ('citeseer', 'Jaccard GCN')             | 0.713 ± 0.011   | 0.659 ± 0.014   | 0.601 ± 0.016    | 0.713 ± 0.011  | 0.654 ± 0.014  | 0.586 ± 0.013   |
| ('citeseer', 'RGCN')                    | 0.646 ± 0.036   | 0.586 ± 0.036   | 0.527 ± 0.037    | 0.646 ± 0.036  | 0.588 ± 0.035  | 0.523 ± 0.036   |
| ('citeseer', 'SVD GCN')                 | 0.646 ± 0.015   | 0.619 ± 0.016   | 0.563 ± 0.020    | 0.646 ± 0.015  | 0.600 ± 0.013  | 0.541 ± 0.021   |
| ('citeseer', 'Soft Medoid GDC (T=0.2)') | 0.703 ± 0.014   | 0.681 ± 0.012   | 0.649 ± 0.012    | 0.703 ± 0.014  | 0.677 ± 0.015  | 0.655 ± 0.013   |
| ('citeseer', 'Soft Medoid GDC (T=0.5)') | 0.716 ± 0.011   | 0.675 ± 0.011   | 0.630 ± 0.011    | 0.716 ± 0.011  | 0.672 ± 0.009  | 0.636 ± 0.007   |
| ('citeseer', 'Soft Medoid GDC (T=1.0)') | 0.711 ± 0.010   | 0.663 ± 0.011   | 0.605 ± 0.014    | 0.711 ± 0.010  | 0.656 ± 0.009  | 0.601 ± 0.007   |
| ('citeseer', 'Vanilla GCN')             | 0.710 ± 0.012   | 0.642 ± 0.014   | 0.570 ± 0.022    | 0.710 ± 0.012  | 0.636 ± 0.009  | 0.556 ± 0.013   |
| ('citeseer', 'Vanilla GDC')             | 0.710 ± 0.009   | 0.635 ± 0.011   | 0.563 ± 0.022    | 0.710 ± 0.009  | 0.622 ± 0.013  | 0.548 ± 0.017   |
| ('cora_ml', 'Jaccard GCN')              | 0.815 ± 0.008   | 0.731 ± 0.003   | 0.660 ± 0.003    | 0.815 ± 0.008  | 0.721 ± 0.005  | 0.625 ± 0.005   |
| ('cora_ml', 'RGCN')                     | 0.807 ± 0.004   | 0.720 ± 0.001   | 0.646 ± 0.004    | 0.807 ± 0.004  | 0.709 ± 0.002  | 0.612 ± 0.005   |
| ('cora_ml', 'SVD GCN')                  | 0.783 ± 0.008   | 0.749 ± 0.006   | 0.677 ± 0.005    | 0.783 ± 0.008  | 0.736 ± 0.008  | 0.641 ± 0.008   |
| ('cora_ml', 'Soft Medoid GDC (T=0.2)')  | 0.806 ± 0.001   | 0.750 ± 0.001   | 0.700 ± 0.004    | 0.806 ± 0.001  | 0.756 ± 0.003  | 0.716 ± 0.002   |
| ('cora_ml', 'Soft Medoid GDC (T=0.5)')  | 0.825 ± 0.003   | 0.749 ± 0.002   | 0.691 ± 0.003    | 0.825 ± 0.003  | 0.748 ± 0.001  | 0.680 ± 0.003   |
| ('cora_ml', 'Soft Medoid GDC (T=1.0)')  | 0.836 ± 0.001   | 0.743 ± 0.001   | 0.680 ± 0.002    | 0.836 ± 0.001  | 0.739 ± 0.002  | 0.657 ± 0.002   |
| ('cora_ml', 'Vanilla GCN')              | 0.823 ± 0.007   | 0.731 ± 0.004   | 0.654 ± 0.004    | 0.823 ± 0.007  | 0.722 ± 0.005  | 0.618 ± 0.006   |
| ('cora_ml', 'Vanilla GDC')              | 0.840 ± 0.002   | 0.735 ± 0.003   | 0.658 ± 0.002    | 0.840 ± 0.002  | 0.726 ± 0.002  | 0.626 ± 0.005   |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

### Certified Robustness

For `Cora ML` and `Citeseer` run
```bash
python script_evaluate_certified_robustness.py
```

which results in:

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributing

This code is licensed under MIT. In you want to contribute, feel free to open a pull request.
