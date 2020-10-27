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


## Requirements

To install requirements: use the subsequent commands. We used Python 3.7.6 and CUDA 10.1. Note that PyTorch Geometric and PyTorch have some version-dependent restriction regarding the supported CUDA versions.

```bash
pip install -r requirements.txt
pip install -r requirements_pytorch_geometric.txt
```

In case you want to use the GPU, you also need to fulfill the [requirements for compiling a custom C++/CUDA extension for PyTorch](https://pytorch.org/tutorials/advanced/cpp_extension.html#using-your-extension). 

You can either build the kernels a priori with
```bash
python install -e ./prebuilt_kernels
```
or PyTorch will try to compile the kernels at runtime.

## Unit Tests

To (unit) test the robust mean functions, you can run (make sure pytest is on your path):

```bash
    pytest .
```

## Training

To train the model in the paper, run one of these commands:

```bash
python experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=1" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=5" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=0.5)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.5}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=1" "model_params={\"label\": \"Soft Medoid GDC (T=0.5)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.5}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=5" "model_params={\"label\": \"Soft Medoid GDC (T=0.5)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.5}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=0.2)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.2}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=1" "model_params={\"label\": \"Soft Medoid GDC (T=0.2)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.2}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=5" "model_params={\"label\": \"Soft Medoid GDC (T=0.2)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.2}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"

python experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"Vanilla GDC\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=1" "model_params={\"label\": \"Vanilla GDC\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=5" "model_params={\"label\": \"Vanilla GDC\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"Vanilla GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=cora_ml" "seed=1" "model_params={\"label\": \"Vanilla GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=cora_ml" "seed=5" "model_params={\"label\": \"Vanilla GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": None}"

python experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"SVD GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": {\"rank\": 50}, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=cora_ml" "seed=1" "model_params={\"label\": \"SVD GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": {\"rank\": 50}, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=cora_ml" "seed=5" "model_params={\"label\": \"SVD GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": {\"rank\": 50}, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"Jaccard GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": {\"threshold\": 0.01}, \"gdc_params\": None}"
python experiment_train.py with "dataset=cora_ml" "seed=1" "model_params={\"label\": \"Jaccard GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": {\"threshold\": 0.01}, \"gdc_params\": None}"
python experiment_train.py with "dataset=cora_ml" "seed=5" "model_params={\"label\": \"Jaccard GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": {\"threshold\": 0.01}, \"gdc_params\": None}"
python experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"RGCN\", \"model\": \"RGCN\", \"n_filters\": 64}"
python experiment_train.py with "dataset=cora_ml" "seed=1" "model_params={\"label\": \"RGCN\", \"model\": \"RGCN\", \"n_filters\": 64}"
python experiment_train.py with "dataset=cora_ml" "seed=5" "model_params={\"label\": \"RGCN\", \"model\": \"RGCN\", \"n_filters\": 64}"
```


```bash
python experiment_train.py with "dataset=citeseer" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=1" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=5" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=0.5)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.5}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=1" "model_params={\"label\": \"Soft Medoid GDC (T=0.5)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.5}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=5" "model_params={\"label\": \"Soft Medoid GDC (T=0.5)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.5}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=0.2)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.2}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=1" "model_params={\"label\": \"Soft Medoid GDC (T=0.2)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.2}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=5" "model_params={\"label\": \"Soft Medoid GDC (T=0.2)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 64, \"temperature\": 0.2}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"

python experiment_train.py with "dataset=citeseer" "seed=0" "model_params={\"label\": \"Vanilla GDC\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=1" "model_params={\"label\": \"Vanilla GDC\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=5" "model_params={\"label\": \"Vanilla GDC\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 64}}"
python experiment_train.py with "dataset=citeseer" "seed=0" "model_params={\"label\": \"Vanilla GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=1" "model_params={\"label\": \"Vanilla GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=5" "model_params={\"label\": \"Vanilla GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": None}"

python experiment_train.py with "dataset=citeseer" "seed=0" "model_params={\"label\": \"SVD GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": {\"rank\": 50}, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=1" "model_params={\"label\": \"SVD GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": {\"rank\": 50}, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=5" "model_params={\"label\": \"SVD GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": {\"rank\": 50}, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=0" "model_params={\"label\": \"Jaccard GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": {\"threshold\": 0.01}, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=1" "model_params={\"label\": \"Jaccard GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": {\"threshold\": 0.01}, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=5" "model_params={\"label\": \"Jaccard GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": {\"threshold\": 0.01}, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=0" "model_params={\"label\": \"RGCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": {\"threshold\": 0.01}, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=1" "model_params={\"label\": \"RGCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": {\"threshold\": 0.01}, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=5" "model_params={\"label\": \"RGCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": {\"threshold\": 0.01}, \"gdc_params\": None}"
python experiment_train.py with "dataset=citeseer" "seed=0" "model_params={\"label\": \"RGCN\", \"model\": \"RGCN\", \"n_filters\": 64}"
python experiment_train.py with "dataset=citeseer" "seed=1" "model_params={\"label\": \"RGCN\", \"model\": \"RGCN\", \"n_filters\": 64}"
python experiment_train.py with "dataset=citeseer" "seed=5" "model_params={\"label\": \"RGCN\", \"model\": \"RGCN\", \"n_filters\": 64}"
```


```bash
python experiment_train.py with "dataset=pubmed" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 32, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=1" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 32, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=5" "model_params={\"label\": \"Soft Medoid GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 32, \"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=0.5)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 32, \"temperature\": 0.5}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=1" "model_params={\"label\": \"Soft Medoid GDC (T=0.5)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 32, \"temperature\": 0.5}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=5" "model_params={\"label\": \"Soft Medoid GDC (T=0.5)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 32, \"temperature\": 0.5}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=0" "model_params={\"label\": \"Soft Medoid GDC (T=0.2)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 32, \"temperature\": 0.2}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=1" "model_params={\"label\": \"Soft Medoid GDC (T=0.2)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 32, \"temperature\": 0.2}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=5" "model_params={\"label\": \"Soft Medoid GDC (T=0.2)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_k_medoid\", \"mean_kwargs\": {\"k\": 32, \"temperature\": 0.2}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"

python experiment_train.py with "dataset=pubmed" "seed=0" "model_params={\"label\": \"Vanilla GDC\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=1" "model_params={\"label\": \"Vanilla GDC\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=5" "model_params={\"label\": \"Vanilla GDC\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.1, \"k\": 32}}"
python experiment_train.py with "dataset=pubmed" "seed=0" "model_params={\"label\": \"Vanilla GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=pubmed" "seed=1" "model_params={\"label\": \"Vanilla GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": None}"
python experiment_train.py with "dataset=pubmed" "seed=5" "model_params={\"label\": \"Vanilla GCN\", \"model\": \"GCN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": None}"
```



```bash
python train.py --dataset=cora_ml --temperature=1 --model-label 'Soft Medoid GDC (T=1.0)'
python train.py --dataset=cora_ml --temperature=0.5 --model-label 'Soft Medoid GDC (T=0.5)'
python train.py --dataset=cora_ml --temperature=0.2 --model-label 'Soft Medoid GDC (T=0.2)'
python train.py --dataset citeseer --temperature 1 --model-label 'Soft Medoid GDC (T=1.0)'
python train.py --dataset=citeseer --temperature=0.5 --model-label 'Soft Medoid GDC (T=0.5)'
python train.py --dataset=citeseer --temperature=0.2 --model-label 'Soft Medoid GDC (T=0.2)'
python train.py --dataset=pubmed --temperature=1 --k=32 --model-label 'Soft Medoid GDC (T=1.0)'
python train.py --dataset=pubmed --temperature=0.5 --k=32 --model-label 'Soft Medoid GDC (T=0.5)'
python train.py --dataset=pubmed --temperature=0.2 --k=32 --model-label 'Soft Medoid GDC (T=0.2)'
# Baselines
python train.py --dataset=cora_ml --disable-gdc --robust-aggregation None --model-label 'Vanilla GCN'
python train.py --dataset=cora_ml --robust-aggregation None --model-label 'Vanilla GDC'
python train.py --dataset=citeseer --disable-gdc --robust-aggregation None --model-label 'Vanilla GCN'
python train.py --dataset=citeseer --robust-aggregation None --model-label 'Vanilla GDC'
python train.py --dataset=pubmed --disable-gdc --robust-aggregation None --model-label 'Vanilla GCN'
python train.py --dataset=pubmed --robust-aggregation None --model-label 'Vanilla GDC'
```

For an overview about the other hyperparameter and all other options invoke:
```bash
python train.py --help
```

## Evaluation

For the evaluation we please execute:
```bash
python attack.py --dataset=cora_ml --attack fgsm
python attack.py --dataset=citeseer --attack fgsm
python attack.py --dataset=cora_ml --attack pgd
python attack.py --dataset=citeseer --attack pgd
```
Similarly for pubmed. However, here you need more than 11 GB of GPU RAM.

## Pre-trained Models

In the [pretrained folder](./pretrained), you find a pretrained model for each dataset (`Cora ML`, `Citeser`, `PubMed`) and the temperatures ![T=0.2](https://latex.codecogs.com/gif.latex?T%3D0.2) and ![T=1](https://latex.codecogs.com/gif.latex?T%3D1).

## Results

Due to the runtime we decided to focus on empirical robustness via the greedy Fast Gradient Signed Method (FGSM) and Projected Gradient Descent (PGD) as presented in Figure 4, Figure 7 and Table 5.

Our model achieves the following performance on:

### Greedy Fast Gradient Signed Method (FGSM)

**Cora ML:**
| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

### Projected Gradient Descent (PGD)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it.

## Contributing

This code is licensed under MIT. In you want to contribute, feel free to open a pull request.
