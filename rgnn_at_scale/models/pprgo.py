from typing import Union

import logging

from datetime import datetime

import torch
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp

from torch.utils.tensorboard import SummaryWriter

from torch_sparse import SparseTensor
from tqdm.auto import tqdm

from rgnn_at_scale.load_ppr import load_ppr
from rgnn_at_scale.data import RobustPPRDataset

from pprgo.pprgo import RobustPPRGoEmmbeddingDiffusions, RobustPPRGo, PPRGoEmmbeddingDiffusions, PPRGo
from pprgo import ppr
from pprgo import utils as ppr_utils


class PPRGoWrapperBase():

    def model_forward(self, *args, **kwargs):
        pass

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

            # TODO: Calculate topk ppr with pytorch so autograd can backprop through adjacency

            if isinstance(adj, SparseTensor):
                num_nodes = adj.size(0)
                adj = adj.to_scipy(layout="csr")
            else:
                # for scipy sparse matrix
                num_nodes = adj.size

            if ppr_idx is None:
                ppr_idx = np.arange(num_nodes)

            # try to read topk test from disk:
            topk_ppr = load_ppr(input_dir=self.ppr_input_dir,
                                dataset=self.dataset,
                                idx=ppr_idx,
                                alpha=self.alpha,
                                eps=self.eps,
                                topk=self.topk,
                                ppr_normalization=self.ppr_normalization,
                                split_desc="test",
                                normalize=self.normalize,
                                make_undirected=self.make_undirected,
                                make_unweighted=self.make_unweighted,
                                shape=(len(ppr_idx), adj.shape[1]))
            if topk_ppr is None:
                topk_ppr = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, ppr_idx,
                                               self.topk,  normalization=self.ppr_normalization)

            # there are to many node for a single forward pass, we need to do batched prediction
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

                logging.info(f"inference batch {batch_id}/{num_batches}")
                logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))
                if device.type == "cuda":
                    logging.info(torch.cuda.max_memory_allocated() / (1024 ** 3))

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
            log_dir="runs",
            # for loading ppr from disk
            ppr_input_dir='/nfs/students/schmidtt/datasets/ppr/papers100M',
            dataset=None,
            normalize=None,
            make_undirected=None,
            make_unweighted=None,
            **kwargs):

        device = next(self.parameters()).device
        self.ppr_input_dir = ppr_input_dir
        self.dataset = dataset
        self.normalize = normalize
        self.make_undirected = make_undirected
        self.make_unweighted = make_unweighted

        logging.info("fit start")
        logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

        if hasattr(self, "mean"):
            mean = self.mean
            temp = self._mean_kwargs["temperature"]
            if hasattr(self, "diffuse_on_embedding"):
                label = "RobustPPRGoDiffEmb"
            else:
                label = "RobustPPRGo"
        else:
            mean = "None"
            temp = 0

            if hasattr(self, "diffuse_on_embedding"):
                label = "VanillaPPRGoDiffEmb"
            else:
                label = "VanillaPPRGo"

        hidden_size = self.hidden_size
        nlayers = self.nlayers
        alpha = self.alpha
        eps = self.eps
        topk = self.topk
        dropout = self.dropout
        ppr_normalization = self.ppr_normalization
        batch_norm = self.mlp.use_batch_norm

        suffix = f"/{label}_alpha{alpha:.0e}_eps{eps:.0e}_topk{topk}_L{nlayers}_H{hidden_size:03d}_D{dropout}_BN{batch_norm}"
        suffix += f"_T{temp:.0e}_{mean}_LR{lr:.0e}_WD{weight_decay:.0e}_S{use_annealing_scheduler:d}_B{batch_size:03d}"
        suffix += f"_N{ppr_normalization}"
        suffix += "_" + str(datetime.now().strftime('%d-%m_%H-%M-%S'))
        writer = SummaryWriter(log_dir=log_dir + suffix)

        if isinstance(adj, SparseTensor):
            adj = adj.to_scipy(layout="csr")

        logging.info(f"tensorboard: {suffix}")
        logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

        # try to read topk train from disk:
        topk_train = load_ppr(input_dir=ppr_input_dir,
                              dataset=dataset,
                              idx=idx_train,
                              alpha=alpha,
                              eps=eps,
                              topk=topk,
                              ppr_normalization=ppr_normalization,
                              split_desc="train",
                              normalize=normalize,
                              make_undirected=make_undirected,
                              make_unweighted=make_unweighted,
                              shape=(len(idx_train), adj.shape[1]))
        if topk_train is None:
            # looks like there was no ppr calculated before hand, so we need to calculate it now
            topk_train = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, idx_train,
                                             self.topk,  normalization=self.ppr_normalization)

        logging.info("topk_train loaded")
        logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

        # try to read topk train from disk:
        topk_val = load_ppr(input_dir=ppr_input_dir,
                            dataset=dataset,
                            idx=idx_val,
                            alpha=alpha,
                            eps=eps,
                            topk=topk,
                            ppr_normalization=ppr_normalization,
                            split_desc="val",
                            normalize=normalize,
                            make_undirected=make_undirected,
                            make_unweighted=make_unweighted,
                            shape=(len(idx_val), adj.shape[1]))
        if topk_val is None:
            topk_val = ppr.topk_ppr_matrix(adj, self.alpha, self.eps, idx_val,
                                           self.topk,  normalization=self.ppr_normalization)

        logging.info("topk_val calculated")
        logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

        train_set = RobustPPRDataset(attr_matrix_all=attr,
                                     ppr_matrix=topk_train,
                                     indices=idx_train,
                                     labels_all=labels,
                                     allow_cache=False)

        logging.info("train_set initalized")
        logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

        val_set = RobustPPRDataset(attr_matrix_all=attr,
                                   ppr_matrix=topk_val,
                                   indices=idx_val,
                                   labels_all=labels,
                                   allow_cache=False)

        logging.info("val_set initalized")
        logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(train_set),
                batch_size=batch_size, drop_last=False
            ),
            batch_size=None,
            num_workers=0,
        )

        logging.info("train_loader initalized")
        logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))

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

                logging.info("Train batch")
                logging.info(ppr_utils.get_max_memory_bytes() / (1024 ** 3))
                if device.type == "cuda":
                    logging.info(torch.cuda.max_memory_allocated() / (1024 ** 3))

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

                writer.add_scalar("Loss/train", loss_train, step)
                writer.add_scalar("Loss/validation", loss_val, step)
                writer.add_scalar("Accuracy/train", train_acc, step)
                writer.add_scalar("Accuracy/validation", val_acc, step)
                writer.add_scalar("Learning Rate", get_lr(optimizer), step)

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
                 n_features: int,
                 n_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 alpha,
                 eps,
                 topk,
                 ppr_normalization,
                 forward_batch_size=128,
                 **kwargs):
        super().__init__(n_features, n_classes, hidden_size, nlayers, dropout, **kwargs)
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
        self.ppr_input_dir = None
        self.dataset = None
        self.normalize = None
        self.make_undirected = None
        self.make_unweighted = None

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
                 n_features: int,
                 n_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 alpha,
                 eps,
                 topk,
                 ppr_normalization,
                 forward_batch_size=128,
                 **kwargs):
        super().__init__(n_features, n_classes, hidden_size, nlayers, dropout, **kwargs)
        self.diffuse_on_embedding = True
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
        self.ppr_input_dir = None
        self.dataset = None
        self.normalize = None
        self.make_undirected = None
        self.make_unweighted = None

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
                 n_features: int,
                 n_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 alpha,
                 eps,
                 topk,
                 ppr_normalization,
                 forward_batch_size=128,
                 mean='soft_k_medoid',
                 **kwargs):
        super().__init__(n_features, n_classes, hidden_size, nlayers, dropout, mean=mean,  **kwargs)
        self.mean = mean
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
        self.ppr_input_dir = None
        self.dataset = None
        self.normalize = None
        self.make_undirected = None
        self.make_unweighted = None

    def forward(self, *args, **kwargs):
        return self.forward_wrapper(*args, **kwargs)

    def model_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class RobustPPRGoDiffEmbWrapper(RobustPPRGoEmmbeddingDiffusions, PPRGoWrapperBase):
    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 hidden_size: int,
                 nlayers: int,
                 dropout: float,
                 alpha,
                 eps,
                 topk,
                 ppr_normalization,
                 forward_batch_size=128,
                 mean='soft_k_medoid',
                 **kwargs):
        super().__init__(n_features, n_classes, hidden_size, nlayers, dropout, mean=mean,  **kwargs)
        self.diffuse_on_embedding = True
        self.mean = mean
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
        self.ppr_input_dir = None
        self.dataset = None
        self.normalize = None
        self.make_undirected = None
        self.make_unweighted = None

    def forward(self, *args, **kwargs):
        return self.forward_wrapper(*args, **kwargs)

    def model_forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
