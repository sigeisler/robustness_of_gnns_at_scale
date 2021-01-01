"""TODO: Do better"""

from copy import deepcopy
import math
import random
from typing import Optional, Union, Tuple

from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch_sparse

from rgnn_at_scale import utils
from rgnn_at_scale.attacks.prbcd import PRBCD
from rgnn_at_scale.models import GCN

FEATURE_MODES = ('symmetric_float', 'binary', 'sparse_pos')


class GANG():

    def __init__(self,
                 adj: torch.sparse.FloatTensor,
                 X: torch.Tensor,
                 labels: torch.Tensor,
                 idx_attack: np.ndarray,
                 model: GCN,
                 device: Union[str, int, torch.device],
                 display_step: int = 20,
                 node_budget: int = 500,
                 edge_budget: int = 500,
                 edge_step_size: int = 10,
                 do_only_connect_test=False,  # TODO: Is not working
                 eps: float = 1e-30,
                 feature_lr: float = 1e-2,
                 feature_init_std: float = 1,
                 feature_max_abs: float = 1,
                 feature_mode: str = 'symmetric_float',  # 'binary', 'sparse_pos'
                 feature_pgd_k: int = 20,
                 feature_greedy_opt: bool = False,
                 do_monitor_time: bool = False,
                 feature_do_use_seeds: bool = False,
                 feature_dedicated_iterations: int = 10,
                 stop_optimizing_if_label_flipped: bool = True,
                 edge_with_random_reverse: bool = True,
                 do_synchronize: bool = False,
                 ** kwargs):
        super().__init__()
        self.device = device
        self.model = deepcopy(model).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.X = X
        self.edge_index = adj.indices()
        self.edge_weight = adj.values()
        self.n = adj.shape[0]
        self.d = X.shape[1]
        self.labels_attack = labels[idx_attack].to(self.device)
        self.idx_attack = idx_attack
        self.display_step = display_step
        self.node_budget = node_budget
        self.edge_budget = edge_budget
        self.edge_step_size = edge_step_size
        self.do_only_connect_test = do_only_connect_test
        self.eps = eps
        self.feature_lr = feature_lr
        self.feature_init_std = feature_init_std
        self.feature_max_abs = feature_max_abs
        assert feature_mode in FEATURE_MODES, f'illegal `feature_mode` {feature_mode}'
        self.feature_mode = feature_mode
        self.feature_pgd_k = feature_pgd_k
        self.feature_greedy_opt = feature_greedy_opt
        self.do_monitor_time = do_monitor_time
        self.feature_do_use_seeds = feature_do_use_seeds
        self.feature_dedicated_iterations = feature_dedicated_iterations
        self.stop_optimizing_if_label_flipped = stop_optimizing_if_label_flipped
        self.edge_with_random_reverse = edge_with_random_reverse
        self.do_synchronize = do_synchronize
        assert self.edge_budget % self.edge_step_size == 0,\
            f'edge budget ({self.edge_budget}) must be dividable by the step size ({self.edge_step_size})'
        self.n_perturbations = 0

    def attack(self, n_perturbations: Optional[int] = None):
        """Perform attack

        Parameters
        ----------
        n_perturbations : int
            Number of edges to be perturbed (assuming an undirected graph)
        """
        assert n_perturbations > self.n_perturbations, (
            f'Number of perturbations must be bigger as this attack is greedy (current {n_perturbations}, '
            f'previous {self.n_perturbations})'
        )
        n_perturbations -= self.n_perturbations
        self.n_perturbations += n_perturbations

        if n_perturbations is None:
            node_budget = self.node_budget
            edge_budget = self.edge_budget
        elif n_perturbations < self.edge_budget:
            node_budget = 1
            edge_budget = ((n_perturbations // self.edge_step_size) + 1) * self.edge_step_size
        else:
            node_budget = (n_perturbations // self.edge_budget) + 1
            edge_budget = self.edge_budget

        features = self.X.to(self.device)
        edge_index = self.edge_index.to(self.device)
        edge_weight = self.edge_weight.to(self.device)

        if self.feature_do_use_seeds:
            n_classes = max(self.labels_attack) + 1
            feature_seeds = [
                features[self.labels_attack[self.labels_attack == c]].mean(0)
                for c
                in range(n_classes)
            ]

        n_features_avg = int(math.ceil(self.X.bool().sum(0).float().mean().item()))

        new_features = None
        for i in tqdm(range(node_budget), desc='Adding nodes'):
            next_node = self.n + i + 1
            if self.do_only_connect_test:
                new_edge_weight = self.eps * torch.ones(len(self.idx_attack)).cuda()
                new_edge_idx = torch.stack([torch.arange(self.n - len(self.idx_attack), self.n),
                                            (next_node - 1) * torch.ones(len(self.idx_attack)).long()]).cuda()
            else:
                new_edge_weight = self.eps * torch.ones(next_node - 1).cuda()
                new_edge_idx = torch.stack([torch.arange(next_node - 1),
                                            (next_node - 1) * torch.ones(next_node - 1).long()]).cuda()

            if self.feature_do_use_seeds:
                seed_id = random.choice(range(n_classes))
                next_new_features = deepcopy(feature_seeds[seed_id])[None, :]
            else:
                next_new_features = self.feature_init_std * torch.randn((1, self.d)).cuda()

            if new_features is not None:
                new_features = torch.cat([new_features.detach(), next_new_features])
            else:
                new_features = next_new_features

            if self.feature_mode == 'symmetric_float':
                new_features_projected = torch.clamp(new_features, -self.feature_max_abs, self.feature_max_abs)
            elif self.feature_mode == 'sparse_pos':
                new_features_projected = torch.clamp(F.dropout(new_features, 1 - n_features_avg / self.d),
                                                     0, self.feature_max_abs)
            else:
                new_features_projected = PRBCD.project(n_features_avg * new_features.size(0), new_features)

            new_features = new_features_projected

            new_edge_weight.requires_grad = True
            new_features.requires_grad = True

            n_steps = edge_budget // self.edge_step_size
            if self.edge_with_random_reverse:
                n_steps += 2

            for j in range(n_steps):
                if self.do_synchronize:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                if self.do_monitor_time:
                    time_start = torch.cuda.Event(enable_timing=True)
                    time_symm = torch.cuda.Event(enable_timing=True)
                    time_forward = torch.cuda.Event(enable_timing=True)
                    time_edge_update = torch.cuda.Event(enable_timing=True)
                    time_feature_update = torch.cuda.Event(enable_timing=True)

                    time_start.record()

                if (
                    not hasattr(self.model, 'do_checkpoint')
                    or not self.model.do_checkpoint
                ):
                    symmetric_edge_index, symmetric_edge_weight = GANG.fuse_adjacency_matrices(
                        edge_index, edge_weight, new_edge_idx, new_edge_weight, m=next_node, n=next_node, op='mean'
                    )
                else:
                    from torch.utils import checkpoint

                    # Due to bottleneck...
                    if len(self.edge_weight) > 100_000_000:
                        symmetric_edge_index, symmetric_edge_weight = GANG.fuse_adjacency_matrices(
                            edge_index.cpu(), edge_weight.cpu(), new_edge_idx.cpu(), new_edge_weight.cpu(),
                            m=next_node, n=next_node, op='mean'
                        )
                        symmetric_edge_index = symmetric_edge_index.to(self.device)
                        symmetric_edge_weight = symmetric_edge_weight.to(self.device)
                    else:
                        # Currently (1.6.0) PyTorch does not support return arguments of `checkpoint` that do not
                        # require gradient. For this reason we need to execute the code twice (due to checkpointing in
                        # fact three times...)
                        with torch.no_grad():
                            symmetric_edge_index = GANG.fuse_adjacency_matrices(
                                edge_index, edge_weight, new_edge_idx, new_edge_weight, m=next_node, n=next_node, op='mean'
                            )[0]

                        symmetric_edge_weight = checkpoint.checkpoint(
                            lambda new_edge_weight: GANG.fuse_adjacency_matrices(
                                edge_index, edge_weight, new_edge_idx, new_edge_weight, m=next_node, n=next_node, op='mean'
                            )[1],
                            new_edge_weight
                        )

                combined_features = torch.cat((features, new_features))

                if self.do_monitor_time:
                    time_symm.record()

                logits = self.model(data=combined_features, adj=(symmetric_edge_index, symmetric_edge_weight))
                not_yet_flipped_mask = logits[self.idx_attack].argmax(-1) == self.labels_attack
                if self.stop_optimizing_if_label_flipped and not_yet_flipped_mask.sum() > 0:
                    loss = F.cross_entropy(logits[self.idx_attack][not_yet_flipped_mask],
                                           self.labels_attack[not_yet_flipped_mask])
                else:
                    loss = F.cross_entropy(logits[self.idx_attack], self.labels_attack)

                if self.do_monitor_time:
                    time_forward.record()

                gradient_edge, gradient_feature = utils.grad_with_checkpoint(
                    loss,
                    [new_edge_weight, new_features],
                )

                if self.edge_with_random_reverse and j == n_steps - 1:
                    edge_step_size = self.edge_step_size + random.choice(range(self.edge_step_size))
                    topk_idx = torch.topk(
                        (self.eps - new_edge_weight) * gradient_edge - (1 - new_edge_weight) * 1e8, edge_step_size
                    )[1]
                    with torch.no_grad():
                        new_edge_weight.index_put_((topk_idx,),
                                                   torch.tensor(self.eps, dtype=torch.float, device=topk_idx.device))
                else:
                    topk_idx = torch.topk((1 - new_edge_weight) * gradient_edge, self.edge_step_size)[1]
                    with torch.no_grad():
                        new_edge_weight.index_put_((topk_idx,),
                                                   torch.tensor(1, dtype=torch.float, device=topk_idx.device))

                if self.do_monitor_time:
                    time_edge_update.record()

                if j > 0 and self.feature_dedicated_iterations is None:
                    with torch.no_grad():
                        new_features = new_features + self.feature_lr * gradient_feature
                        new_features = torch.clamp(new_features, -self.feature_max_abs, self.feature_max_abs)
                    new_features.requires_grad = True

                if self.do_monitor_time:
                    time_feature_update.record()
                    torch.cuda.synchronize()

                    print(f'Symmetrize took: {time_start.elapsed_time(time_symm)}')
                    print(f'Forward took: {time_symm.elapsed_time(time_forward)}')
                    print(f'Edge update took: {time_forward.elapsed_time(time_edge_update)}')
                    print(f'Feature update took: {time_edge_update.elapsed_time(time_feature_update)}')

                if j % self.display_step == 0:
                    accuracy = (
                        logits.argmax(-1)[self.idx_attack] == self.labels_attack
                    ).float().mean()
                    print(f'Adversarial node {i} after adding {j + 1} edges, the accuracy is {100 * accuracy:.3f} %')

            new_edge_idx = new_edge_idx[:, new_edge_weight == 1]
            new_edge_weight = new_edge_weight[new_edge_weight == 1]

            symmetric_edge_index, symmetric_edge_weight = GANG.fuse_adjacency_matrices(
                edge_index, edge_weight, new_edge_idx, new_edge_weight.detach(), m=next_node, n=next_node, op='max'
            )

            if self.do_synchronize:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if self.feature_dedicated_iterations is not None:
                optimizer = torch.optim.Adam((new_features,), lr=self.feature_lr)
                accuracy = (
                    logits.argmax(-1)[self.idx_attack] == self.labels_attack
                ).float().mean()
                print(f'Adversarial node {i} before optimizing the features, we have an accuracy '
                      f'of {100 * accuracy:.3f} %')
                for j in range(self.feature_dedicated_iterations):
                    if self.feature_mode == 'symmetric_float':
                        new_features_projected = torch.clamp(new_features, -self.feature_max_abs, self.feature_max_abs)
                    elif self.feature_mode == 'sparse_pos':
                        new_features_projected = torch.clamp(new_features, 0, self.feature_max_abs)
                    else:
                        new_features_projected = PRBCD.project(n_features_avg * new_features.size(0), new_features)

                    new_features.data.copy_(new_features_projected)
                    combined_features = torch.cat((features, new_features))

                    logits = self.model(data=combined_features, adj=(symmetric_edge_index, symmetric_edge_weight))
                    loss = F.cross_entropy(logits[self.idx_attack], self.labels_attack)

                    optimizer.zero_grad()
                    (-loss).backward()
                    optimizer.step()

                if self.feature_mode == 'symmetric_float':
                    new_features = torch.clamp(new_features, -self.feature_max_abs, self.feature_max_abs).detach()
                elif self.feature_mode == 'sparse_pos':
                    new_features_projected = torch.clamp(new_features, 0, self.feature_max_abs)
                else:
                    s = PRBCD.project(n_features_avg * new_features.size(0), new_features.detach())
                    s[s <= self.eps] = 0
                    s = s.cpu().numpy()

                    best_loss = float('-Inf')
                    best_accuracy = float('Inf')
                    while best_loss == float('-Inf'):
                        for _ in range(self.feature_pgd_k):
                            sampled = np.random.binomial(1, s)

                            new_features_projected = torch.tensor(
                                sampled, dtype=torch.float, device=new_features.device)
                            combined_features = torch.cat((features, new_features_projected))
                            logits = self.model(
                                data=combined_features,
                                adj=(symmetric_edge_index, symmetric_edge_weight)
                            )
                            loss = F.cross_entropy(logits[self.idx_attack], self.labels_attack)
                            accuracy = (
                                logits.argmax(-1)[self.idx_attack] == self.labels_attack
                            ).float().mean()
                            if accuracy < best_accuracy:
                                best_loss = loss
                                best_accuracy = accuracy
                                best_features = new_features_projected.clone()
                    new_features = best_features

                logits = self.model(
                    data=torch.cat((features, new_features)),
                    adj=(symmetric_edge_index, symmetric_edge_weight)
                )
                loss = F.cross_entropy(logits[self.idx_attack], self.labels_attack)
                accuracy = (
                    logits.argmax(-1)[self.idx_attack] == self.labels_attack
                ).float().mean()
                print(f'Adversarial node {i} after optimizing the features, we have an accuracy '
                      f'of {100 * accuracy:.3f} %')

            if self.feature_greedy_opt:
                features = torch.cat((features, new_features)).detach()
                new_features = None

            edge_index = symmetric_edge_index.detach()
            edge_weight = symmetric_edge_weight.detach()

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.n = self.n + i + 1
        self.adj_adversary = torch.sparse.FloatTensor(
            edge_index, edge_weight, (self.n, self.n)
        ).coalesce().detach()
        if not self.feature_greedy_opt:
            self.attr_adversary = torch.cat((features, new_features)).detach()
        else:
            self.attr_adversary = features.detach()

        self.X = self.attr_adversary.clone()
        self.edge_index = edge_index.clone()
        self.edge_weight = edge_weight.clone()

        assert self.n == self.X.shape[0]

    @staticmethod
    def fuse_adjacency_matrices(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                                modified_edge_index: torch.Tensor, modified_edge_weight_diff: torch.Tensor,
                                n: int, m: int, op: str = 'sum') -> Tuple[torch.Tensor, torch.Tensor]:
        modified_edge_index, modified_edge_weight = utils.to_symmetric(
            modified_edge_index, modified_edge_weight_diff, n
        )
        edge_index = torch.cat((edge_index, modified_edge_index), dim=-1)
        edge_weight = torch.cat((edge_weight, modified_edge_weight))

        # FIXME: This seems to be the current bottle neck. Maybe change to merge of sorted lists
        edge_index, edge_weight = torch_sparse.coalesce(edge_index, edge_weight, m=n, n=n, op='sum')
        return edge_index, edge_weight