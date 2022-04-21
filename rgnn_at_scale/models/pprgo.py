from typing import Any, Dict, Union, List
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch import nn
from torch_sparse import SparseTensor
from torch_scatter import scatter
from tqdm.auto import tqdm

from rgnn_at_scale.data import RobustPPRDataset
from rgnn_at_scale.aggregation import ROBUST_MEANS
from rgnn_at_scale.helper import utils
from rgnn_at_scale.helper import ppr_utils as ppr

patch_typeguard()


class PPRGoMLP(nn.Module):
    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 n_filters: List[int],
                 dropout: float,
                 batch_norm: bool = False):
        super().__init__()
        self.use_batch_norm = batch_norm

        layers = []
        n_filter_last_layer = n_features
        for n_filter in n_filters:
            layers.append(nn.Linear(n_filter_last_layer, n_filter, bias=False))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(n_filter))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            n_filter_last_layer = n_filter

        layers.append(nn.Linear(n_filter_last_layer, n_classes, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, X):

        embs = self.layers(X)
        return embs


class PPRGo(nn.Module):
    """
    The vanilla PPRGo Model of Bojchevski & Klicpera et al.
    The implementation was taken from https://github.com/TUM-DAML/pprgo_pytorch

    @inproceedings{bojchevski2020pprgo,
        title={Scaling Graph Neural Networks with Approximate PageRank},
        author={Bojchevski, Aleksandar and Klicpera, Johannes and Perozzi, Bryan and Kapoor, Amol and Blais,
                Martin and R{\'o}zemberczki, Benedek and Lukasik, Michal and G{\"u}nnemann, Stephan},
        booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
        year={2020},
        publisher={ACM},
        address={New York, NY, USA},
    }

    Parameters
    ----------
    n_features : int
        Number of attributes for each node
    n_classes : int
        Number of classes for prediction
    n_filters : List[int]
        number of dimensions for the hidden units of each layer.
    dropout : int
        Dropout rate between 0 and 1
    batch_norm : bool, optional
        If true use batch norm in every layer block between the linearity and activation function, by default False
    aggr : str, optional
        The reduce operation to be used in the message passing step to aggregate all incoming node messages
        Possible values are "sum", "mean","min" or "max". (default: "sum")
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 n_filters: List[int],
                 dropout: float,
                 batch_norm: bool = False,
                 aggr: str = "sum",
                 **kwargs):
        super().__init__()
        self.mlp = PPRGoMLP(n_features, n_classes, n_filters, dropout, batch_norm)
        self.aggr = aggr

    @typechecked
    def forward(self,
                X: TensorType["num_ppr_nodes", "n_features"],
                ppr_scores: TensorType["num_ppr_nodes"],
                ppr_idx: TensorType["num_ppr_nodes"]) -> TensorType["batch_size", "n_classes"]:
        """
        Parameters:
            X: torch_sparse.SparseTensor of shape (num_ppr_nodes, n_features)
                The node features for all nodes which were assigned a ppr score
            ppr_scores: torch.Tensor of shape (num_ppr_nodes)
                The ppr scores are calculate for every node of the batch individually.
                This tensor contains these flattend ppr scores for every node in the batch.
            ppr_idx: torch.Tensor of shape (num_ppr_nodes)
                The id of the batch that the corresponding ppr_score entry belongs to

        Returns:
            propagated_logits: torch.Tensor of shape (batch_size, n_classes)

        """
        # logits of shape (num_batch_nodes, n_classes)
        logits = self.mlp(X)
        propagated_logits = scatter(logits * ppr_scores[:, None], ppr_idx[:, None],
                                    dim=0, dim_size=ppr_idx[-1] + 1, reduce=self.aggr)
        return propagated_logits


class RobustPPRGo(nn.Module):
    """
    The robust version of the PPRGo Model of Bojchevski & Klicpera et al
    which was extended to include the robust aggregation functions:
    - soft_k_medoid
    - soft_medoid (not scalable)
    - k_medoid
    - medoid (not scalable)
    - dimmedian

    The core implementation of PPRGo was taken from https://github.com/TUM-DAML/pprgo_pytorch

    @inproceedings{bojchevski2020pprgo,
    title={Scaling Graph Neural Networks with Approximate PageRank},
    author={Bojchevski, Aleksandar and Klicpera, Johannes and Perozzi, Bryan and Kapoor, Amol and Blais,
            Martin and R{\'o}zemberczki, Benedek and Lukasik, Michal and G{\"u}nnemann, Stephan},
    booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
    year={2020},
    publisher={ACM},
    address={New York, NY, USA},
    }

    Parameters
    ----------
    n_features : int
        Number of attributes for each node
    n_classes : int
        Number of classes for prediction
    n_filters : List[int]
        number of dimensions for the hidden units of each layer.
    dropout : int
        Dropout rate between 0 and 1
    batch_norm : bool, optional
        If true use batch norm in every layer block between the linearity and activation function, by default False
    mean : str, optional
        The desired mean (see above for the options), by default 'soft_k_medoid'
    mean_kwargs : Dict[str, Any], optional
        Arguments for the mean, by default dict(k=64, temperature=1.0, with_weight_correction=True)
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 n_filters: List[int],
                 dropout: float,
                 batch_norm: bool = False,
                 mean='soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=64,
                                                    temperature=1.0,
                                                    with_weight_correction=True),
                 **kwargs):
        super().__init__()
        self._mean = ROBUST_MEANS[mean]
        self._mean_kwargs = mean_kwargs
        self.mlp = PPRGoMLP(n_features, n_classes, n_filters, dropout, batch_norm)

    def forward(self,
                X: SparseTensor,
                ppr_scores: SparseTensor):
        """
        Parameters:
            X: torch_sparse.SparseTensor of shape (n_neighbors, n_features)
                The node features of all neighboring from nodes of the ppr_matrix (training nodes)
            ppr_matrix: torch_sparse.SparseTensor of shape (n_neighbors, num_nodes)
                The sparse personalized pagerank matrix for all neighbors contained in the feature matrix.

        Returns:
            propagated_logits: torch.Tensor of shape (batch_size, n_classes)

        """
        # logits of shape (num_batch_nodes, n_classes)
        logits = self.mlp(X)

        if self._mean.__name__ == 'soft_median' and ppr_scores.size(0) == 1 and 'temperature' in self._mean_kwargs:
            c = logits.shape[1]
            weights = ppr_scores.storage.value()
            with torch.no_grad():
                sort_idx = logits.argsort(0)
                weights_cumsum = weights[sort_idx].cumsum(0)
                median_idx = sort_idx[(weights_cumsum < weights_cumsum[-1][None, :] / 2).sum(0), torch.arange(c)]
            median = logits[median_idx, torch.arange(c)]
            distances = torch.norm(logits - median[None, :], dim=1) / pow(c, 1 / 2)

            soft_weights = weights * F.softmax(-distances / self._mean_kwargs['temperature'], dim=-1)
            soft_weights /= soft_weights.sum()
            new_logits = (soft_weights[:, None] * weights.sum() * logits).sum(0)

            return new_logits[None, :]

        if "k" in self._mean_kwargs.keys() and "with_weight_correction" in self._mean_kwargs.keys():
            # `n` less than `k` and `with_weight_correction` is not implemented
            # so we need to make sure we set with_weight_correction to false if n less than k
            if self._mean_kwargs["k"] > X.size(0):
                logging.info("no with_weight_correction")
                return self._mean(ppr_scores,
                                  logits,
                                  # we can not manipluate self._mean_kwargs because this would affect
                                  # the next call to forward, so we do it this way
                                  with_weight_correction=False,
                                  ** {k: v for k, v in self._mean_kwargs.items() if k != "with_weight_correction"})
        return self._mean(ppr_scores,
                          logits,
                          **self._mean_kwargs)


class PPRGoWrapperBase(ABC):
    """
        The base class for PPRGo wrapper classes defining
            1) default hyperparameter values
            2) the custom training procedure of PPRGo models
            3) a general wrapper around pprgos forward function, calculating the
            approximate page rank matrix from the adjacency if ommited in the forward call
    """

    @typechecked
    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 n_filters: Union[int, List[int]] = 512,
                 n_layers: int = 4,
                 dropout: float = 0.0,
                 alpha: float = 0.1,
                 eps: float = 1e-3,
                 topk: int = 64,
                 ppr_normalization: str = "row",
                 forward_batch_size: int = 128,
                 batch_norm: bool = False,
                 mean: str = 'soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=64,
                                                    temperature=1.0,
                                                    with_weight_correction=True),
                 ppr_cache_params: Dict[str, Any] = None,
                 **kwargs):
        """
        Parameters
        ----------
        n_features : int
            Number of attributes for each node
        n_classes : int
            Number of classes for prediction
        n_filters : Union[int, List[int]]
            number of dimensions for the hidden units.
            Either a single integer for all layers or a list of integers to specify the hidden units
            for each layer individually. If a list of integers is given, the n_layers parameter is ignored
        n_layers : int
            number of layers before the message passing step (via graph diffusion)
        dropout : int
            Dropout rate between 0 and 1
        batch_norm : bool, optional
            If true use batch norm in every layer block between the linearity and activation function, by default False
        mean : str, optional
            The desired mean (see above for the options), by default 'soft_k_medoid'
        mean_kwargs : Dict[str, Any], optional
            Arguments for the mean, by default dict(k=64, temperature=1.0, with_weight_correction=True)
        mean : str, optional
            The desired mean (see above for the options), by default 'soft_k_medoid'
        forward_batch_size: int, optional
            In case the forward method does not recieve ppr_scores, this argument specifies how large the batches
            will be that are processed at once in a single forward pass.
        alpha: int, optional
            The alpha value (restart probability) that is used to calculate the approximate topk ppr matrix
        eps: int, optional
            The threshold used as stopping criterion for the iterative approximation algorithm used for the ppr matrix
        topk: int, optional
            The top k elements to keep in each row of the ppr matrix.
        ppr_normalization: int, optional
            The normalization that is applied to the top k ppr matrix before passing it to the PPRGo model.
            Possible values are 'sym', 'col' and 'row' (by default 'row')

        ppr_cache_params: Dict[str, any]
            To allow for caching the ppr matrix on the hard drive and loading it from disk the following keys in the
            dictionary need to provide the necessary information:
                data_artifact_dir : str
                    The folder name/path in which to look for the storage (TinyDB) objects
                data_storage_type : str
                    The name of the storage (TinyDB) table name that's supposed to be used for caching ppr matrices
                dataset : str
                    The name of the dataset for which this model will be applied. This is necessary to make sure the
                    correct ppr matrix is loaded from the disk for conscutive calls
                make_directed : bool
                    Wether the dataset passed to this model will be a directed graph or not. Necessary for the same
                    reason as the dataset name

        """
        self.n_features = n_features
        if isinstance(n_filters, list):
            self.n_filters = n_filters
        elif isinstance(n_filters, int):
            self.n_filters = [n_filters] * (n_layers - 1)
        else:
            raise TypeError("n_filters must be integer or list of integers")

        self.n_classes = n_classes
        self.dropout = dropout
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.ppr_normalization = ppr_normalization
        self.forward_batch_size = forward_batch_size
        self.batch_norm = batch_norm
        self.mean = mean
        self.mean_kwargs = mean_kwargs
        self.ppr_cache_params = ppr_cache_params

    @abstractmethod
    def model_forward(self, *args, **kwargs):
        pass

    def release_cache(self):
        self.ppr_cache_params = None

    @typechecked
    def forward_wrapper(self,
                        attr: TensorType["n_nodes", "n_classes"],
                        adj: Union[SparseTensor, sp.csr_matrix, TensorType["n_nodes", "n_nodes"]] = None,
                        ppr_scores: SparseTensor = None,
                        ppr_idx=None):
        """
        Wrapper around the forward function of PPRGo models.
        Fully (auto)-differentiable only iff ppr_scores is not None!
        If the ppr_scores are not given, they will be calculated on the fly or loaded from cache (disk)

        Parameters
        ----------
        attr : Torch.Tensor
            The feature/attribute matrix of shape (n_nodes, n_features)
        adj : Union[SparseTensor, sp.csr_matrix],
            The adjacency matrix used for calculating the personalized page rank matrix.
            Should be of shape (n_nodes, n_nodes)
        ppr_scores : SparseTensor
            The precalculated personalized page rank matrix
        ppr_idx: np.Array
            The list of node ids for which the personalized page rank matrix should be calculated from the adjacency

        """

        device = next(self.parameters()).device

        if isinstance(adj, torch.Tensor):
            adj = SparseTensor.from_dense(adj.cpu()).to(device)

        if ppr_scores is not None:

            source_idx, neighbor_idx, ppr_vals = ppr_scores.coo()
            ppr_matrix = ppr_scores[:, neighbor_idx.unique()]
            attr_matrix = attr[neighbor_idx.unique()]

            return self.model_forward(attr_matrix.to(device), ppr_matrix.to(device))
        else:

            # we need to precompute the ppr_score first

            if isinstance(adj, SparseTensor):
                adj = adj.to_scipy(layout="csr")

            num_nodes = adj.shape[0]

            if ppr_idx is None:
                ppr_idx = np.arange(num_nodes)

            # try to read topk test from storage:
            topk_ppr = None
            if self.ppr_cache_params is not None:
                # late import as a workaround to avoid circular import issue
                from rgnn_at_scale.helper.io import Storage
                storage = Storage(self.ppr_cache_params["data_artifact_dir"])
                params = dict(dataset=self.ppr_cache_params["dataset"],
                              alpha=self.alpha,
                              ppr_idx=np.array(ppr_idx),
                              eps=self.eps,
                              topk=self.topk,
                              ppr_normalization=self.ppr_normalization,
                              make_undirected=self.ppr_cache_params["make_undirected"])

                stored_topk_ppr = storage.find_sparse_matrix(self.ppr_cache_params["data_storage_type"],
                                                             params, find_first=True)
                topk_ppr, _ = stored_topk_ppr[0] if len(stored_topk_ppr) == 1 else (None, None)

            if topk_ppr is None:
                topk_ppr = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, ppr_idx.copy(),
                                               self.topk,  normalization=self.ppr_normalization)

                # save topk_ppr to disk
                if self.ppr_cache_params is not None:
                    params["ppr_idx"] = np.array(ppr_idx)
                    storage.save_sparse_matrix(self.ppr_cache_params["data_storage_type"], params,
                                               topk_ppr, ignore_duplicate=True)

            # there are usually to many nodes for a single forward pass, we need to do batched prediction
            data_set = RobustPPRDataset(
                attr_matrix_all=attr,
                ppr_matrix=topk_ppr,
                indices=ppr_idx,
                allow_cache=False)
            data_loader = torch.utils.data.DataLoader(
                dataset=data_set,
                sampler=torch.utils.data.BatchSampler(
                    torch.utils.data.SequentialSampler(data_set),
                    batch_size=self.forward_batch_size, drop_last=False
                ),
                batch_size=None,
                num_workers=0,
            )
            num_predictions = topk_ppr.shape[0]

            logits = torch.zeros(num_predictions, self.n_classes, device="cpu", dtype=torch.float32)

            num_batches = len(data_loader)
            display_step = max(int(num_batches / 10), 1)
            for batch_id, (idx, xbs, _) in enumerate(data_loader):

                if batch_id % display_step == 0:
                    logging.info(f"Memory Usage before inference batch {batch_id}/{num_batches}:")
                    logging.info(utils.get_max_memory_bytes() / (1024 ** 3))
                    if device.type == "cuda":
                        logging.info(torch.cuda.max_memory_allocated() / (1024 ** 3))

                xbs = [xb.to(device) for xb in xbs]
                start = batch_id * self.forward_batch_size
                end = start + xbs[1].size(0)  # batch_id * batch_size
                logits[start:end] = self.model_forward(*xbs).cpu()

            return logits

    @typechecked
    def fit(self,
            adj: Union[SparseTensor, sp.csr_matrix],
            attr: TensorType["n_nodes", "n_classes"],
            labels: TensorType["n_nodes"],
            idx_train: np.ndarray,
            idx_val: np.ndarray,
            lr: float,
            weight_decay: float,
            patience: int,
            use_annealing_scheduler: bool = False,
            scheduler_warm_restarts: bool = True,
            annealing_scheduler_T_0: int = 3,
            scheduler_time: str = "epoch",
            scheduler_step: int = 20,
            optim: str = "Adam",
            max_epochs: int = 200,
            batch_size: int = 512,
            batch_mult_val: int = 4,
            eval_step: int = 1,
            display_step: int = 50,
            # for loading ppr from disk
            ppr_cache_params: dict = None,
            ** kwargs):
        device = next(self.parameters()).device

        if ppr_cache_params is not None:
            # update ppr_cache_params
            self.ppr_cache_params = ppr_cache_params

        if isinstance(adj, SparseTensor):
            adj = adj.to_scipy(layout="csr")

        topk_train = None
        # try to read topk test from storage:
        if self.ppr_cache_params is not None:
            # late import as a workaround to avoid circular import issue
            from rgnn_at_scale.helper.io import Storage
            storage = Storage(self.ppr_cache_params["data_artifact_dir"])
            params = dict(dataset=self.ppr_cache_params["dataset"],
                          alpha=self.alpha,
                          ppr_idx=np.array(idx_train),
                          eps=self.eps,
                          topk=self.topk,
                          ppr_normalization=self.ppr_normalization,
                          make_undirected=self.ppr_cache_params["make_undirected"])

            stored_topk_train = storage.find_sparse_matrix(self.ppr_cache_params["data_storage_type"],
                                                           params, find_first=True)
            topk_train, _ = stored_topk_train[0] if len(stored_topk_train) == 1 else (None, None)

        if topk_train is None:
            # looks like there was no ppr calculated before hand, so we need to calculate it now
            topk_train = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, idx_train.copy(),
                                             self.topk,  normalization=self.ppr_normalization)
            # save topk_ppr to disk
            if self.ppr_cache_params is not None:
                params["ppr_idx"] = np.array(idx_train)
                storage.save_sparse_matrix(self.ppr_cache_params["data_storage_type"], params,
                                           topk_train, ignore_duplicate=True)

        logging.info("Memory Usage after calculating/loading topk ppr for train:")
        logging.info(utils.get_max_memory_bytes() / (1024 ** 3))

        # try to read topk train from disk:
        topk_val = None
        if self.ppr_cache_params is not None:
            params["ppr_idx"] = np.array(idx_val)

            stored_topk_val = storage.find_sparse_matrix(self.ppr_cache_params["data_storage_type"],
                                                         params, find_first=True)
            topk_val, _ = stored_topk_val[0] if len(stored_topk_val) == 1 else (None, None)

        if topk_val is None:
            topk_val = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, idx_val.copy(),
                                           self.topk,  normalization=self.ppr_normalization)
            # save topk_ppr to disk
            if self.ppr_cache_params is not None:
                params["ppr_idx"] = np.array(idx_val)
                storage.save_sparse_matrix(self.ppr_cache_params["data_storage_type"], params,
                                           topk_val, ignore_duplicate=True)

        logging.info("Memory Usage after calculating/loading topk ppr for validation:")
        logging.info(utils.get_max_memory_bytes() / (1024 ** 3))

        train_set = RobustPPRDataset(attr_matrix_all=attr,
                                     ppr_matrix=topk_train,
                                     indices=idx_train,
                                     labels_all=labels,
                                     allow_cache=False)

        val_set = RobustPPRDataset(attr_matrix_all=attr,
                                   ppr_matrix=topk_val,
                                   indices=idx_val,
                                   labels_all=labels,
                                   allow_cache=False)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(train_set),
                batch_size=batch_size, drop_last=False
            ),
            batch_size=None,
            num_workers=0,
        )

        trace_train_loss = []
        trace_val_loss = []
        trace_train_acc = []
        trace_val_acc = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        if optim == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:  # use adam
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        if use_annealing_scheduler:
            if scheduler_warm_restarts:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, annealing_scheduler_T_0)
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, max_epochs)

        best_epoch_loss = np.inf
        num_batches = len(train_loader)
        step = 0
        epoch_pbar = tqdm(range(max_epochs), desc='Training Epoch...')
        for it in epoch_pbar:
            epoch_loss_val = 0
            epoch_acc_val = 0
            epoch_acc_train = 0

            for batch_train_idx, xbs, yb in train_loader:
                xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

                # logging.info(f"Memory Usage before training batch {step}:")
                # logging.info(utils.get_max_memory_bytes() / (1024 ** 3))
                # if device.type == "cuda":
                #     logging.info(torch.cuda.max_memory_allocated() / (1024 ** 3))

                loss_train, ncorrect_train = self.__run_batch(xbs, yb, optimizer, train=True)

                train_acc = ncorrect_train / float(yb.shape[0])

                # validation on batch of val_set
                val_batch_size = batch_mult_val * batch_size
                rnd_idx = np.random.choice(len(val_set), size=len(val_set), replace=False)[:val_batch_size]
                batch_val_idx, xbs, yb = val_set[rnd_idx]
                xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

                loss_val, ncorrect_val = self.__run_batch(xbs, yb, None, train=False)
                val_acc = ncorrect_val / float(yb.shape[0])

                epoch_loss_val += loss_val / num_batches
                epoch_acc_val += val_acc / num_batches
                epoch_acc_train += train_acc / num_batches

                trace_train_loss.append(loss_train)
                trace_val_loss.append(loss_val)
                trace_train_acc.append(train_acc)
                trace_val_acc.append(val_acc)

                if use_annealing_scheduler and scheduler_time == "batch":
                    if step % scheduler_step == 0:
                        logging.info("Scheduler Batch Step CosineAnnealingWarmRestarts\n")
                        scheduler.step()

                step += 1

            epoch_pbar.set_description(f"Training Epoch... acc_train: {epoch_acc_train: .4f},"
                                       f"acc_val: {epoch_acc_val: .4f}", refresh=False)

            if use_annealing_scheduler and scheduler_time == "epoch":
                logging.info("Scheduler Epoch Step CosineAnnealingWarmRestarts\n")
                scheduler.step()

            if epoch_loss_val < best_epoch_loss:
                best_epoch_loss = epoch_loss_val
                best_epoch = it
                best_state = {key: value.cpu() for key, value in self.state_dict().items()}
                # logging.info(f"Save best_state for new best_epoch_loss {best_epoch_loss}\n")
            else:
                if it >= best_epoch + patience:
                    logging.info("Early stopping due to increase in validation loss")
                    break
                # logging.info(f"No decrease in validation loss in epoch {it} since best epoch {best_epoch} ...")

            # restore the best validation state
        self.load_state_dict(best_state)
        return {"loss": trace_val_loss, "acc": trace_val_acc}, {"loss": trace_train_loss, "acc": trace_train_acc}

    @typechecked
    def __run_batch(self, xbs: list, yb: TensorType["batch_size"], optimizer, train: bool):
        # Set model to training mode
        if train:
            self.train()
        else:
            self.eval()

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(train):
            logits = self.model_forward(*xbs)
            loss = F.cross_entropy(logits, yb)
            top1 = torch.argmax(logits, dim=1)
            ncorrect = torch.sum(top1 == yb)

            # backward + optimize only if in training phase
            if train:
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item(), ncorrect.detach().cpu().item()


class PPRGoWrapper(PPRGo, PPRGoWrapperBase):
    """
        Wrapper class around the Vanilla PPRGo model.
        Use this class to instantiate a PPRGo model that includes the calculation and caching of
        the ppr matrix as well as the training procedure.

    """

    def __init__(self,
                 *args,
                 **kwargs):
        # using the constructor of the wrapper base class to set/validate the required/optional model params
        PPRGoWrapperBase.__init__(self, *args, **kwargs)

        PPRGo.__init__(self, self.n_features, self.n_classes, self.n_filters, self.dropout,
                       batch_norm=self.batch_norm, mean=self.mean, mean_kwargs=self.mean_kwargs)

    def forward(self, *args, **kwargs):
        return self.forward_wrapper(*args, **kwargs)

    @typechecked
    def model_forward(self,
                      attr: TensorType["n_nodes", "n_features"],
                      ppr_matrix: SparseTensor,
                      **kwargs):
        source_idx, neighbor_idx, ppr_scores = ppr_matrix.coo()
        attr = attr[neighbor_idx]
        return super().forward(attr, ppr_scores, source_idx)


class RobustPPRGoWrapper(RobustPPRGo, PPRGoWrapperBase):
    def __init__(self,
                 *args,
                 **kwargs):
        PPRGoWrapperBase.__init__(self, *args, **kwargs)
        RobustPPRGo.__init__(self, self.n_features, self.n_classes, self.n_filters, self.dropout,
                             batch_norm=self.batch_norm, mean=self.mean, mean_kwargs=self.mean_kwargs)

    def forward(self, *args, **kwargs):
        return self.forward_wrapper(*args, **kwargs)

    def model_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
