from typing import Any, Dict, Union

import math
import logging

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


class PPRGoMLP(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 batch_norm: bool = False):
        super().__init__()
        self.use_batch_norm = batch_norm

        layers = [nn.Linear(num_features, hidden_size, bias=False)]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_size))

        for i in range(nlayers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))

        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, num_classes, bias=False))

        self.layers = nn.Sequential(*layers)

    def forward(self, X):

        embs = self.layers(X)
        return embs

    def reset_parameters(self):
        self.layers.reset_parameters()


class PPRGo(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 batch_norm: bool = False,
                 aggr: str = "sum",
                 **kwargs):
        super().__init__()
        self.mlp = PPRGoMLP(num_features, num_classes,
                            hidden_size, nlayers, dropout, batch_norm)
        self.aggr = aggr

    def forward(self,
                X: SparseTensor,
                ppr_scores: torch.Tensor,
                ppr_idx: torch.Tensor):
        """
        Parameters:
            X: torch_sparse.SparseTensor of shape (num_ppr_nodes, num_features)
                The node features for all nodes which were assigned a ppr score
            ppr_scores: torch.Tensor of shape (num_ppr_nodes)
                The ppr scores are calculate for every node of the batch individually.
                This tensor contains these concatenated ppr scores for every node in the batch.
            ppr_idx: torch.Tensor of shape (num_ppr_nodes)
                The id of the batch that the corresponding ppr_score entry belongs to

        Returns:
            propagated_logits: torch.Tensor of shape (batch_size, num_classes)

        """
        # logits of shape (num_batch_nodes, num_classes)
        logits = self.mlp(X)
        propagated_logits = scatter(logits * ppr_scores[:, None], ppr_idx[:, None],
                                    dim=0, dim_size=ppr_idx[-1] + 1, reduce=self.aggr)
        return propagated_logits


class PPRGoEmmbeddingDiffusions(nn.Module):
    """
    Just like PPRGo, but diffusing/aggregating on the embedding space and not the logit space.
    """

    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 batch_norm: bool = False,
                 skip_connection=False,
                 aggr: str = "sum",
                 **kwargs):
        super().__init__()
        # TODO: rewrite PPRGoMLP such that it doesn't expect at least n_layers >= 2.
        self.skip_connection = skip_connection

        layer_num_mlp = math.ceil(nlayers / 2)
        layer_num_mlp_logits = math.floor(nlayers / 2)

        if self.skip_connection:
            assert hidden_size > num_features, "hidden size must be greater than num_features for this skip_connection implementation to work"
            self.mlp = PPRGoMLP(num_features, hidden_size - num_features,
                                hidden_size, layer_num_mlp, dropout, batch_norm)
        else:
            self.mlp = PPRGoMLP(num_features, hidden_size,
                                hidden_size, layer_num_mlp, dropout, batch_norm)

        self.mlp_logits = PPRGoMLP(hidden_size, num_classes,
                                   hidden_size, layer_num_mlp_logits, dropout, batch_norm)
        self.aggr = aggr

    def forward(self,
                X: SparseTensor,
                ppr_scores: torch.Tensor,
                ppr_idx: torch.Tensor):
        """
        Parameters:
            X: torch_sparse.SparseTensor of shape (num_ppr_nodes, num_features)
                The node features for all nodes which were assigned a ppr score
            ppr_scores: torch.Tensor of shape (num_ppr_nodes)
                The ppr scores are calculate for every node of the batch individually.
                This tensor contains these concatenated ppr scores for every node in the batch.
            ppr_idx: torch.Tensor of shape (num_ppr_nodes)
                The id of the batch that the corresponding ppr_score entry belongs to

        Returns:
            propagated_logits: torch.Tensor of shape (batch_size, num_classes)

        """
        # logits of shape (num_batch_nodes, num_classes)
        embedding = self.mlp(X)
        propagated_embedding = scatter(embedding * ppr_scores[:, None], ppr_idx[:, None],
                                       dim=0, dim_size=ppr_idx[-1] + 1, reduce=self.aggr)
        if self.skip_connection:
            # concatenated node features and propagated node embedding on feature dimension:
            propagated_embedding = torch.cat((X[ppr_idx.unique()], propagated_embedding), dim=-1)

        return self.mlp_logits(propagated_embedding)


class RobustPPRGo(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 batch_norm: bool = False,
                 mean='soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=32,
                                                    temperature=1.0,
                                                    with_weight_correction=True),
                 **kwargs):
        super().__init__()
        self._mean = ROBUST_MEANS[mean]
        self._mean_kwargs = mean_kwargs
        self.mlp = PPRGoMLP(num_features, num_classes,
                            hidden_size, nlayers, dropout, batch_norm)

    def forward(self,
                X: SparseTensor,
                ppr_scores: SparseTensor):
        """
        Parameters:
            X: torch_sparse.SparseTensor of shape (num_ppr_nodes, num_features)
                The node features of all neighboring from nodes of the ppr_matrix (training nodes)
            ppr_matrix: torch_sparse.SparseTensor of shape (ppr_num_nonzeros, num_features)
                The node features of all neighboring nodes of the training nodes in
                the graph derived from the Personal Page Rank as specified by idx

        Returns:
            propagated_logits: torch.Tensor of shape (batch_size, num_classes)

        """
        # logits of shape (num_batch_nodes, num_classes)
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
                print("no with_weight_correction")
                return self._mean(ppr_scores,
                                  logits,
                                  # we can not manipluate self._mean_kwargs because this would affect
                                  # the next call to forward, so we do it this way
                                  with_weight_correction=False,
                                  ** {k: v for k, v in self._mean_kwargs.items() if k != "with_weight_correction"})
        return self._mean(ppr_scores,
                          logits,
                          **self._mean_kwargs)


class RobustPPRGoEmmbeddingDiffusions(nn.Module):
    """
    Just like RobustPPRGo, but diffusing/aggregating on the embedding space and not the logit space.
    """

    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 batch_norm: bool = False,
                 mean='soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=32,
                                                    temperature=1.0,
                                                    with_weight_correction=True),
                 **kwargs):
        super().__init__()
        # TODO: rewrite PPRGoMLP such that it doesn't expect at least n_layers >= 2.
        assert nlayers >= 4, "nlayers must be 4 or greater for this implementation to work"
        self._mean = ROBUST_MEANS[mean]
        self._mean_kwargs = mean_kwargs
        self.mlp = PPRGoMLP(num_features, hidden_size,
                            hidden_size, nlayers - 2, dropout, batch_norm)

        self.mlp_logits = PPRGoMLP(hidden_size, num_classes,
                                   hidden_size, 2, dropout, batch_norm)

    def forward(self,
                X: SparseTensor,
                ppr_scores: SparseTensor):
        """
        Parameters:
            X: torch_sparse.SparseTensor of shape (num_ppr_nodes, num_features)
                The node features of all neighboring from nodes of the ppr_matrix (training nodes)
            ppr_matrix: torch_sparse.SparseTensor of shape (ppr_num_nonzeros, num_features)
                The node features of all neighboring nodes of the training nodes in
                the graph derived from the Personal Page Rank as specified by idx

        Returns:
            propagated_logits: torch.Tensor of shape (batch_size, num_classes)

        """
        # logits of shape (num_batch_nodes, num_classes)
        embedding = self.mlp(X)

        if self._mean.__name__ == 'soft_median' and ppr_scores.size(0) == 1 and 'temperature' in self._mean_kwargs:
            c = embedding.shape[1]
            weights = ppr_scores.storage.value()
            with torch.no_grad():
                sort_idx = embedding.argsort(0)
                weights_cumsum = weights[sort_idx].cumsum(0)
                median_idx = sort_idx[(weights_cumsum < weights_cumsum[-1][None, :] / 2).sum(0), torch.arange(c)]
            median = embedding[median_idx, torch.arange(c)]
            distances = torch.norm(embedding - median[None, :], dim=1) / pow(c, 1 / 2)

            soft_weights = weights * F.softmax(-distances / self._mean_kwargs['temperature'], dim=-1)
            soft_weights /= soft_weights.sum()
            new_embedding = (soft_weights[:, None] * weights.sum() * embedding).sum(0)

            diffused_embedding = new_embedding[None, :]

        elif "k" in self._mean_kwargs.keys() and "with_weight_correction" in self._mean_kwargs.keys() \
                and self._mean_kwargs["k"] > X.size(0):
            # `n` less than `k` and `with_weight_correction` is not implemented
            # so we need to make sure we set with_weight_correction to false if n less than k
            print("no with_weight_correction")
            diffused_embedding = self._mean(ppr_scores,
                                            embedding,
                                            # we can not manipluate self._mean_kwargs because this would affect
                                            # the next call to forward, so we do it this way
                                            with_weight_correction=False,
                                            ** {k: v for k, v in self._mean_kwargs.items() if k != "with_weight_correction"})
        else:
            diffused_embedding = self._mean(ppr_scores,
                                            embedding,
                                            **self._mean_kwargs)
        return self.mlp_logits(diffused_embedding)


class PPRGoWrapperBase():

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 hidden_size: int = 512,
                 nlayers: int = 4,
                 dropout: float = 0.0,
                 alpha: float = 0.1,
                 eps: float = 1e-3,
                 topk: int = 64,
                 ppr_normalization: str = "row",
                 forward_batch_size: int = 128,
                 batch_norm: bool = False,
                 skip_connection: bool = False,
                 mean: str = 'soft_k_medoid',
                 mean_kwargs: Dict[str, Any] = dict(k=32,
                                                    temperature=1.0,
                                                    with_weight_correction=True),
                 ppr_cache_params: Dict[str, Any] = None,
                 **kwargs):
        """
        ppr_cache_params: dict
            data_artifact_dir : str
            data_storage_type : str
            dataset : str
            normalize : str
            make_unweighted : bool
            make_unweighted : bool
        """
        self.num_features = n_features
        self.num_classes = n_classes
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.dropout = dropout
        self.alpha = alpha
        self.eps = eps
        self.topk = topk
        self.ppr_normalization = ppr_normalization
        self.forward_batch_size = forward_batch_size
        self.batch_norm = batch_norm
        self.skip_connection = skip_connection
        self.mean = mean
        self.mean_kwargs = mean_kwargs
        self.ppr_cache_params = ppr_cache_params

    def model_forward(self, *args, **kwargs):
        pass

    def release_cache(self):
        self.ppr_cache_params = None

    def forward_wrapper(self,
                        attr: torch.Tensor,
                        adj: Union[SparseTensor, sp.csr_matrix],
                        ppr_scores: SparseTensor = None,
                        ppr_idx=None):

        device = next(self.parameters()).device
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
                              ppr_idx=list(map(int, ppr_idx)),
                              eps=self.eps,
                              topk=self.topk,
                              ppr_normalization=self.ppr_normalization,
                              normalize=self.ppr_cache_params["normalize"],
                              make_undirected=self.ppr_cache_params["make_undirected"],
                              make_unweighted=self.ppr_cache_params["make_unweighted"])

                stored_topk_ppr = storage.find_sparse_matrix(self.ppr_cache_params["data_storage_type"],
                                                             params, find_first=True)
                topk_ppr, _ = stored_topk_ppr[0] if len(stored_topk_ppr) == 1 else (None, None)

            if topk_ppr is None:
                topk_ppr = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, ppr_idx,
                                               self.topk,  normalization=self.ppr_normalization)

                # save topk_ppr to disk
                if self.ppr_cache_params is not None:
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

            logits = torch.zeros(num_predictions, self.num_classes, device="cpu", dtype=torch.float32)

            num_batches = len(data_loader)
            for batch_id, (idx, xbs, _) in enumerate(data_loader):

                logging.debug(f"Memory Usage before inference batch {batch_id}/{num_batches}:")
                logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))
                if device.type == "cuda":
                    logging.debug(torch.cuda.max_memory_allocated() / (1024 ** 3))

                xbs = [xb.to(device) for xb in xbs]
                start = batch_id * self.forward_batch_size
                end = start + xbs[1].size(0)  # batch_id * batch_size
                logits[start:end] = self.model_forward(*xbs).cpu()

            return logits

    def fit(self,
            adj: Union[SparseTensor, sp.csr_matrix],
            attr: torch.Tensor,
            labels: torch.Tensor,
            idx_train: np.ndarray,
            idx_val: np.ndarray,
            lr,
            weight_decay: int,
            patience: int,
            use_annealing_scheduler=False,
            scheduler_warm_restarts=True,
            annealing_scheduler_T_0=3,
            scheduler_time="epoch",
            scheduler_step=20,
            optim="Adam",
            max_epochs: int = 200,
            batch_size=512,
            batch_mult_val=4,
            eval_step=1,
            display_step=50,
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
                          ppr_idx=list(map(int, idx_train)),
                          eps=self.eps,
                          topk=self.topk,
                          ppr_normalization=self.ppr_normalization,
                          split_desc="train",
                          normalize=self.ppr_cache_params["normalize"],
                          make_undirected=self.ppr_cache_params["make_undirected"],
                          make_unweighted=self.ppr_cache_params["make_unweighted"])

            stored_topk_train = storage.find_sparse_matrix(self.ppr_cache_params["data_storage_type"],
                                                           params, find_first=True)
            topk_train, _ = stored_topk_train[0] if len(stored_topk_train) == 1 else (None, None)

        if topk_train is None:
            # looks like there was no ppr calculated before hand, so we need to calculate it now
            topk_train = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, idx_train,
                                             self.topk,  normalization=self.ppr_normalization)
            # save topk_ppr to disk
            if self.ppr_cache_params is not None:
                storage.save_sparse_matrix(self.ppr_cache_params["data_storage_type"], params,
                                           topk_train, ignore_duplicate=True)

        logging.debug(f"Memory Usage after calculating/loading topk ppr for train:")
        logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

        # try to read topk train from disk:
        topk_val = None
        if self.ppr_cache_params is not None:
            params["ppr_idx"] = list(map(int, idx_val))
            params["split_desc"] = "val"

            stored_topk_val = storage.find_sparse_matrix(self.ppr_cache_params["data_storage_type"],
                                                         params, find_first=True)
            topk_val, _ = stored_topk_val[0] if len(stored_topk_val) == 1 else (None, None)

        if topk_val is None:
            topk_val = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, idx_val,
                                           self.topk,  normalization=self.ppr_normalization)
            # save topk_ppr to disk
            if self.ppr_cache_params is not None:
                storage.save_sparse_matrix(self.ppr_cache_params["data_storage_type"], params,
                                           topk_val, ignore_duplicate=True)

        logging.debug(f"Memory Usage after calculating/loading topk ppr for validation:")
        logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))

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

        best_loss = np.inf

        step = 0
        epoch_pbar = tqdm(range(max_epochs), desc='Training Epoch...')
        for it in epoch_pbar:
            batch_pbar = tqdm(train_loader, desc="Training Batch...")
            for batch_train_idx, xbs, yb in batch_pbar:
                xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

                logging.debug(f"Memory Usage before training batch {step}:")
                logging.debug(utils.get_max_memory_bytes() / (1024 ** 3))
                if device.type == "cuda":
                    logging.debug(torch.cuda.max_memory_allocated() / (1024 ** 3))

                loss_train, ncorrect_train = self.__run_batch(xbs, yb, optimizer, train=True)

                train_acc = ncorrect_train / float(yb.shape[0])

                # validation on batch of val_set
                val_batch_size = batch_mult_val * batch_size
                rnd_idx = np.random.choice(len(val_set), size=len(val_set), replace=False)[:val_batch_size]
                batch_val_idx, xbs, yb = val_set[rnd_idx]
                xbs, yb = [xb.to(device) for xb in xbs], yb.to(device)

                loss_val, ncorrect_val = self.__run_batch(xbs, yb, None, train=False)
                val_acc = ncorrect_val / float(yb.shape[0])

                trace_train_loss.append(loss_train)
                trace_val_loss.append(loss_val)
                trace_train_acc.append(train_acc)
                trace_val_acc.append(val_acc)

                if loss_val < best_loss:
                    best_loss = loss_val
                    best_epoch = it
                    best_state = {key: value.cpu() for key, value in self.state_dict().items()}
                    logging.info(f"Save best_state for new best_loss {best_loss}\n")
                else:
                    if it >= best_epoch + patience:
                        break

                batch_pbar.set_description(f"Epoch: {it:}, loss_train: {loss_train: .5f}, loss_val: {loss_val: .5f}",
                                           refresh=False)
                if use_annealing_scheduler and scheduler_time == "batch":
                    if step % scheduler_step == 0:
                        logging.info("Scheduler Batch Step CosineAnnealingWarmRestarts\n")
                        scheduler.step()

                step += 1

            epoch_pbar.set_description(f"Training Epoch... acc_train: {train_acc: .4f}, acc_val: {val_acc: .4f}",
                                       refresh=False)

            if use_annealing_scheduler and scheduler_time == "epoch":
                logging.info("Scheduler Epoch Step CosineAnnealingWarmRestarts\n")
                scheduler.step()

            # restore the best validation state
        self.load_state_dict(best_state)
        return {"loss": trace_val_loss, "acc": trace_val_acc}, {"loss": trace_train_loss, "acc": trace_train_acc}

    def __run_batch(self, xbs, yb, optimizer, train):
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
    def __init__(self,
                 *args,
                 **kwargs):
        PPRGoWrapperBase.__init__(self, *args, **kwargs)
        PPRGo.__init__(self, self.num_features, self.num_classes,
                       self.hidden_size, self.nlayers, self.dropout,
                       batch_norm=self.batch_norm, skip_connection=self.skip_connection,
                       mean=self.mean, mean_kwargs=self.mean_kwargs)

    def forward(self, *args, **kwargs):
        return self.forward_wrapper(*args, **kwargs)

    def model_forward(self,
                      attr: torch.Tensor,
                      ppr_matrix: SparseTensor,
                      **kwargs):
        source_idx, neighbor_idx, ppr_scores = ppr_matrix.coo()
        attr = attr[neighbor_idx]
        return super().forward(attr, ppr_scores, source_idx)


class PPRGoDiffEmbWrapper(PPRGoEmmbeddingDiffusions, PPRGoWrapperBase):
    def __init__(self,
                 *args,
                 **kwargs):
        PPRGoWrapperBase.__init__(self, *args, **kwargs)
        PPRGoEmmbeddingDiffusions.__init__(self, self.num_features, self.num_classes,
                                           self.hidden_size, self.nlayers, self.dropout,
                                           batch_norm=self.batch_norm, skip_connection=self.skip_connection,
                                           mean=self.mean, mean_kwargs=self.mean_kwargs)

    def forward(self, *args, **kwargs):
        return self.forward_wrapper(*args, **kwargs)

    def model_forward(self,
                      attr: torch.Tensor,
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
        RobustPPRGo.__init__(self, self.num_features, self.num_classes,
                             self.hidden_size, self.nlayers, self.dropout,
                             batch_norm=self.batch_norm, skip_connection=self.skip_connection,
                             mean=self.mean, mean_kwargs=self.mean_kwargs)

    def forward(self, *args, **kwargs):
        return self.forward_wrapper(*args, **kwargs)

    def model_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class RobustPPRGoDiffEmbWrapper(RobustPPRGoEmmbeddingDiffusions, PPRGoWrapperBase):

    def __init__(self,
                 *args,
                 **kwargs):
        PPRGoWrapperBase.__init__(self, *args, **kwargs)
        RobustPPRGoEmmbeddingDiffusions.__init__(self, self.num_features, self.num_classes,
                                                 self.hidden_size, self.nlayers, self.dropout,
                                                 batch_norm=self.batch_norm, skip_connection=self.skip_connection,
                                                 mean=self.mean, mean_kwargs=self.mean_kwargs)

    def forward(self, *args, **kwargs):
        return self.forward_wrapper(*args, **kwargs)

    def model_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
