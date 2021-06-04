# Reliable Graph Neural Networks via Robust Aggregation

## Structure

Besides the standard python artifacts we provide:

- `cache`: for the pretrained models / attacked adjacency matrices
- `config`: the configuration files grouped by experiments
- `datasets`: for storing the datasets
- `experiments`: source code defining the types of experiments
- `kernels`: the custom kernel package
- `notebooks`: for (jupyter) notebooks
- `rgnn_at_scale`: the source code
- `tests`: unit tests for some important parts of the code
- `script_execute_experiment.py`: the _main script_ to execute an experiment

## Installation

*Note: The setup is tested only for Linux 18.04 and will likely not work on other platforms.*

For simplicity we recommend to install PyTorch a priori via anaconda:
```bash
conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
We used Python 3.7.6 and CUDA 10.2. We provide custom CUDA kernels that are fairly simple implementations for a `row-wise topk` and `row-wise weighted median` on a sparse matrix.

Due to custom CUDA kernels, you must be able to compile via `nvcc`. Conda handles the c++ compiler etc. You also must have installed the CUDA toolkit and should select the matching CUDA version for your environment. Note that PyTorch Geometric and PyTorch have some version-dependent restriction regarding the supported CUDA versions. See also [Build PyTorch from source](https://pytorch.org/get-started/locally/#mac-from-source) which captures the requirements for building custom extensions. 

### Main Package

Thereafter we can install the actual module via (alternatively use `python install .`):
```bash
pip install -r requirements.txt
pip install .
```
By default the requirements are installed with very restrictive versioning since we did not test any other configuration. If you have version conflicts, you can also build without version restrictions via omitting the command `pip install -r requirements.txt` (not tested).

### Prebuilt Kernels

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

As a minimum working example we provide the jupyter notebook (`conda install jupyter`): `notebooks/Figure - Which nodes get attacked?.ipynb`

This is the code used to analyze the learning curves and the distribution of attacked nodes (e.g. Fig. 2).

## Training

*Note: after open sourcing we will provide the full collection of pretrained models and in the case of transfer attacks we will also provide all perturbed adjacency matrices. For now we only include the artifacts for Cora ML.*

For the training and evaluation code we decided to provide SEML/Sacred experiments which make it very easy to run the same code from the command line or on your cluster.

To train the models you can use the `script_execute_experiment` script and simply specif the respective configuration (if the configuration specifies `partition: gpu_large` you need at least 32 GB of GPU memory):
```bash
python script_execute_experiment.py --config-files 'config/train/cora_and_citeseer.yaml'
```

Alternatively, you can also execute the experiment directly passing the desired configuration:
```bash
python experiments/experiment_train.py with "dataset=cora_ml" "seed=0" "model_params={\"label\": \"Soft Median GDC (T=1.0)\", \"model\": \"RGNN\", \"do_cache_adj_prep\": True, \"n_filters\": 64, \"dropout\": 0.5, \"mean\": \"soft_median\", \"mean_kwargs\": {\"temperature\": 1.0}, \"svd_params\": None, \"jaccard_params\": None, \"gdc_params\": {\"alpha\": 0.15, \"k\": 64}}" "artifact_dir=cache" "binary_attr=False"
```

## Evaluation

For evaluation, we use the locally stored models in the `cache` folder (unless specified differently).

### Empirical Robustness

Similarly to training, we provide a script that runs the attacks for different seeds for all pretrained models (here for a local attack on Cora ML and Citeseer using PR-BCD):
```bash
python script_execute_experiment.py --config-files 'config/attack_evasion_local_direct/cora_and_citeseer_localprbcd.yaml'
```
