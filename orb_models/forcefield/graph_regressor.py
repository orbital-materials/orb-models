from typing import Dict, Literal, Optional, Tuple, Union

import numpy
import torch
import torch.nn as nn

from orb_models.forcefield import base, segment_ops
from orb_models.forcefield.gns import _KEY, MoleculeGNS
from orb_models.forcefield.nn_util import build_mlp
from orb_models.forcefield.property_definitions import PROPERTIES, PropertyDefinition
from orb_models.forcefield.reference_energies import REFERENCE_ENERGIES

global HAS_WARNED_FOR_TF32_MATMUL
HAS_WARNED_FOR_TF32_MATMUL = False


def warn_for_tf32_matmul():
    """Warn the user once only if they are not using tensorfloat matmuls."""
    global HAS_WARNED_FOR_TF32_MATMUL
    if (
        not HAS_WARNED_FOR_TF32_MATMUL
        and torch.cuda.is_available()
        and not torch.get_float32_matmul_precision() == "high"
    ):
        print(
            "Warning! You are using a model on the GPU without enabling tensorfloat matmuls."
            "This can be up to 2x slower than enabling this flag."
            "Enable it with torch.set_float32_matmul_precision('high')"
        )
        HAS_WARNED_FOR_TF32_MATMUL = True
        print(f"Current matmul precision is: {torch.get_float32_matmul_precision()}")


class LinearReferenceEnergy(torch.nn.Module):
    """Linear reference energy (no bias term)."""

    def __init__(
        self,
        weight_init: Optional[numpy.ndarray] = None,
        trainable: Optional[bool] = None,
    ):
        super().__init__()

        if trainable is None:
            trainable = weight_init is None

        self.linear = torch.nn.Linear(118, 1, bias=False)
        if weight_init is not None:
            self.linear.weight.data = torch.tensor(weight_init, dtype=torch.float32)
        if not trainable:
            self.linear.weight.requires_grad = False

    def forward(self, atom_types: torch.Tensor, n_node: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LinearReferenceEnergy.

        Args:
            atom_types: A tensor of atomic numbers of shape (n_atoms,)

        Returns:
            A tensor of shape (n_graphs, 1) containing the reference energy.
        """
        one_hot_atomic = torch.nn.functional.one_hot(atom_types, num_classes=118).type(
            torch.float32
        )
        reduced = segment_ops.aggregate_nodes(one_hot_atomic, n_node, reduction="sum")
        return self.linear(reduced)


class ScalarNormalizer(torch.nn.Module):
    """Scalar normalizer that learns mean and std from data.

    NOTE: Multi-dimensional tensors are flattened before updating
    the running mean/std. This is desired behaviour for force targets.
    """

    def __init__(
        self,
        init_mean: Optional[Union[torch.Tensor, float]] = None,
        init_std: Optional[Union[torch.Tensor, float]] = None,
        init_num_batches: Optional[int] = 1000,
    ) -> None:
        """Initializes the ScalarNormalizer.

        To enhance training stability, consider setting an init mean + std.
        """
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(1, affine=False, momentum=None)  # type: ignore
        if init_mean is not None:
            assert init_std is not None
            self.bn.running_mean = torch.tensor([init_mean])
            self.bn.running_var = torch.tensor([init_std**2])
            self.bn.num_batches_tracked = torch.tensor([init_num_batches])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize by running mean and std."""
        if self.training:
            self.bn(x.view(-1, 1))
        return (x - self.bn.running_mean) / torch.sqrt(self.bn.running_var)  # type: ignore

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the forward normalization."""
        return x * torch.sqrt(self.bn.running_var) + self.bn.running_mean  # type: ignore


class NodeHead(torch.nn.Module):
    """Node prediction head that can be appended to a base model.

    This head could be added to the foundation model to enable
    auxiliary tasks during pretraining, or added afterwards
    during a finetuning step.
    """

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        target: Union[str, PropertyDefinition],
        dropout: Optional[float] = None,
        remove_mean: bool = True,
        remove_torque_for_nonpbc_systems: bool = True,
    ):
        """Initializes the NodeHead MLP.

        Args:
            input_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            target: either the name of a PropertyDefinition or a PropertyDefinition itself.
            dropout: The level of dropout to apply.
            remove_mean: Whether to remove the mean of the node features.
            remove_torque_for_nonpbc_systems: Whether to remove net torque from the 
                force predictions for non-PBC systems.
        """
        super().__init__()
        if isinstance(target, str):
            target = PROPERTIES[target]
        self.target_property = target

        if target.domain != "real":
            raise ValueError("NodeHead only supports real targets.")

        if target.means is not None and target.stds is not None:
            self.normalizer = ScalarNormalizer(
                init_mean=target.means,
                init_std=target.stds,
            )
        else:
            self.normalizer = ScalarNormalizer()

        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=self.target_property.dim,
            dropout=dropout,
        )

        self.remove_mean = remove_mean
        self.remove_torque_for_nonpbc_systems = remove_torque_for_nonpbc_systems

    def forward(self, batch: base.AtomGraphs) -> base.AtomGraphs:
        """Predictions with raw logits (no sigmoid/softmax or any inverse transformations)."""
        feat = batch.node_features[_KEY]
        pred = self.mlp(feat)

        if self.remove_mean:
            system_means = segment_ops.aggregate_nodes(
                pred, batch.n_node, reduction="mean"
            )
            node_broadcasted_means = torch.repeat_interleave(
                system_means, batch.n_node, dim=0
            )
            pred = pred - node_broadcasted_means

        if self.remove_torque_for_nonpbc_systems:
            pred = selectively_remove_net_torque_for_nonpbc_systems(
                pred, batch.positions, batch.system_features["cell"], batch.n_node
            )

        batch.node_features["node_pred"] = pred
        return batch

    def predict(self, batch: base.AtomGraphs) -> torch.Tensor:
        """Predict node/edge/graph attribute."""
        out = self(batch)
        pred = out.node_features["node_pred"]
        return self.normalizer.inverse(pred)

    def loss(self, batch: base.AtomGraphs):
        """Apply mlp to compute loss and metrics."""
        batch_n_node = batch.n_node
        assert batch.node_targets is not None
        target = batch.node_targets[self.target_property.name].squeeze(-1)
        pred = batch.node_features["node_pred"].squeeze(-1)
        # make sure we remove fixed atoms before normalization
        pred, target, batch_n_node = _remove_fixed_atoms(
            pred, target, batch_n_node, batch.fix_atoms, self.training
        )
        mae = torch.abs(pred - self.normalizer(target))
        raw_pred = self.normalizer.inverse(pred)
        raw_mae = torch.abs(raw_pred - target)

        if self.target_property.dim > 1:
            mae = mae.mean(dim=-1)
            mae = segment_ops.aggregate_nodes(
                mae, batch_n_node, reduction="mean"
            ).mean()
            raw_mae = raw_mae.mean(dim=-1)
            raw_mae = segment_ops.aggregate_nodes(
                raw_mae, batch_n_node, reduction="mean"
            ).mean()
        else:
            mae = mae.mean()
            raw_mae = raw_mae.mean()
        metrics = {
            "node_mae": mae.item(),
            "node_mae_raw": raw_mae.item(),
            "node_cosine_sim": torch.cosine_similarity(raw_pred, target, dim=-1)
            .mean()
            .item(),
            "fwt_0.03": forces_within_threshold(raw_pred, target, batch_n_node),
        }
        return base.ModelOutput(loss=mae, log=metrics)


class GraphHead(torch.nn.Module):
    """MLP Regression head that can be appended to a base model.

    This head could be added to the foundation model to enable
    auxiliary tasks during pretraining, or added afterwards
    during a finetuning step.
    """

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        target: Union[str, PropertyDefinition],
        node_aggregation: Literal["sum", "mean"] = "mean",
        dropout: Optional[float] = None,
        compute_stress: Optional[bool] = False,
    ):
        """Initializes the GraphHead MLP.

        Args:
            input_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            target: either the name of a PropertyDefinition or a PropertyDefinition itself
            node_aggregation: The method for aggregating the node features
                from the pretrained model representations.
            dropout: The level of dropout to apply.
        """
        super().__init__()
        if isinstance(target, str):
            target = PROPERTIES[target]
        self.target_property = target

        if target.means is not None and target.stds is not None:
            self.normalizer = ScalarNormalizer(
                init_mean=target.means,
                init_std=target.stds,
            )
        else:
            self.normalizer = ScalarNormalizer()

        self.node_aggregation = node_aggregation
        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=self.target_property.dim,
            dropout=dropout,
        )
        activation_dict = {
            "real": torch.nn.Identity,
            "binary": torch.nn.Sigmoid,
            "categorical": torch.nn.Softmax,
        }
        self.output_activation = activation_dict[self.target_property.domain]()
        self.compute_stress = compute_stress

    def forward(self, batch: base.AtomGraphs) -> base.AtomGraphs:
        """Predictions with raw logits (no sigmoid/softmax or any inverse transformations)."""
        feat = batch.node_features[_KEY]

        # aggregate to get a tensor of shape (num_graphs, latent_dim)
        input = segment_ops.aggregate_nodes(
            feat,
            batch.n_node,
            reduction=self.node_aggregation,
        )
        pred = self.mlp(input)
        if self.compute_stress:
            # we need to name the stress prediction differently
            batch.system_features["stress_pred"] = pred
        else:
            batch.system_features["graph_pred"] = pred
        return batch

    def predict(self, batch: base.AtomGraphs) -> torch.Tensor:
        """Predict node/edge/graph attribute."""
        pred = self(batch)
        if self.compute_stress:
            logits = pred.system_features["stress_pred"].squeeze(-1)
        else:
            logits = pred.system_features["graph_pred"].squeeze(-1)
        probs = self.output_activation(logits)
        if self.target_property.domain == "real":
            probs = self.normalizer.inverse(probs)
        return probs

    def loss(self, batch: base.AtomGraphs):
        """Apply mlp to compute loss and metrics.

        Depending on whether the target is real/binary/categorical, we
        use an MSE/cross-entropy loss. In the case of cross-entropy, the
        preds are logits (not normalised) to take advantage of numerically
        stable log-softmax.
        """
        assert batch.system_targets is not None
        target = batch.system_targets[self.target_property.name].squeeze(-1)
        if self.compute_stress:
            pred = batch.system_features["stress_pred"].squeeze(-1)
        else:
            pred = batch.system_features["graph_pred"].squeeze(-1)

        domain = self.target_property.domain
        # Short circuit for binary and categorical targets
        if domain == "binary":
            loss, metrics = bce_loss(pred, target, "graph")
            return base.ModelOutput(loss=loss, log=metrics)
        if domain == "categorical":
            loss, metrics = cross_entropy_loss(pred, target, "graph")
            return base.ModelOutput(loss=loss, log=metrics)

        normalized_target = self.normalizer(target)
        errors = normalized_target - pred
        mae = torch.abs(errors).mean()

        raw_pred = self.normalizer.inverse(pred)
        raw_mae = torch.abs(raw_pred - target).mean()
        if self.compute_stress:
            metrics = {"stress_mae": mae.item(), "stress_mae_raw": raw_mae.item()}
        else:
            metrics = {"graph_mae": mae.item(), "graph_mae_raw": raw_mae.item()}
        return base.ModelOutput(loss=mae, log=metrics)


class EnergyHead(GraphHead):
    """Energy prediction head that can be appended to a base model."""

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        target: Union[str, PropertyDefinition] = "energy",
        predict_atom_avg: bool = True,
        reference_energy_name: str = "mp-traj-d3",  # or 'vasp-shifted'
        train_reference: bool = False,
        dropout: Optional[float] = None,
        node_aggregation: Optional[str] = None,
    ):
        """Initializes the EnergyHead MLP.

        Args:
            input_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            target: either the name of a PropertyDefinition or a PropertyDefinition itself.
            predict_atom_avg: Whether to predict the average atom energy or total.
            reference_energy_name: The name of the linear reference energy model to use.
            train_reference: Whether the reference energy params are learnable.
            dropout: The level of dropout to apply.
            node_aggregation: (deprecated) The method for aggregating the node features
        """
        ref = REFERENCE_ENERGIES[reference_energy_name]
        target = PROPERTIES[target] if isinstance(target, str) else target
        if predict_atom_avg:
            assert node_aggregation or "mean" == "mean"
            target.means = torch.tensor([ref.residual_mean_per_atom])
            target.stds = torch.tensor([ref.residual_std_per_atom])
        else:
            assert node_aggregation or "sum" == "sum"
            target.means = torch.tensor([ref.residual_mean])
            target.stds = torch.tensor([ref.residual_std])

        super().__init__(
            latent_dim=latent_dim,
            num_mlp_layers=num_mlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            target=target,
            node_aggregation=node_aggregation,  # type: ignore
            dropout=dropout,
        )
        self.reference = LinearReferenceEnergy(
            weight_init=ref.coefficients, trainable=train_reference
        )
        self.atom_avg = predict_atom_avg

    def predict(self, batch: base.AtomGraphs) -> torch.Tensor:
        """Predict energy."""
        pred = self(batch).system_features["graph_pred"]
        pred = self.normalizer.inverse(pred).squeeze(-1)
        if self.atom_avg:
            pred = pred * batch.n_node
        pred = pred + self.reference(batch.atomic_numbers, batch.n_node)
        return pred

    def loss(self, batch: base.AtomGraphs):
        """Apply mlp to compute loss and metrics."""
        assert batch.system_targets is not None
        target = batch.system_targets[self.target_property.name].squeeze(-1)
        pred = batch.system_features["graph_pred"].squeeze(-1)

        reference = self.reference(batch.atomic_numbers, batch.n_node)
        reference_target = target - reference
        if self.atom_avg:
            reference_target = reference_target / batch.n_node

        normalized_reference = self.normalizer(reference_target)
        model_loss = normalized_reference - pred

        raw_pred = self.normalizer.inverse(pred)
        if self.atom_avg:
            raw_pred = raw_pred * batch.n_node
        raw_mae = torch.abs((raw_pred + reference) - target).mean()

        reference_mae = torch.abs(reference_target).mean()
        model_mae = torch.abs(model_loss).mean()
        metrics = {
            "energy_reference_mae": reference_mae.item(),
            "energy_mae": model_mae.item(),
            "energy_mae_raw": raw_mae.item(),
        }
        return base.ModelOutput(loss=model_mae, log=metrics)


class GraphRegressor(nn.Module):
    """Graph Regressor for finetuning.

    The GraphRegressor combines a pretrained base model
    with two regression heads for finetuning; one head for
    a node level task, and one for a graph level task.
    The base model can be optionally fine-tuned.
    The regression head is a linear/MLP transformation
    of the sum/avg of the graph's node activations.
    """

    def __init__(
        self,
        model: MoleculeGNS,
        node_head: Optional[NodeHead] = None,
        graph_head: Optional[GraphHead] = None,
        stress_head: Optional[GraphHead] = None,
        model_requires_grad: bool = True,
        cutoff_layers: Optional[int] = None,
    ) -> None:
        """Initializes the GraphRegressor.

        Args:
            node_head : The regression head to use for node prediction.
            graph_head: The regression head to use for graph prediction.
            model: An optional pre-constructed, pretrained model to use for transfer learning/finetuning.
            model_requires_grad: Whether the underlying model should
                be finetuned or not.
        """
        super().__init__()

        if (node_head is None) and (graph_head is None):
            raise ValueError("Must provide at least one node/graph head.")
        self.node_head = node_head
        self.graph_head = graph_head
        self.stress_head = stress_head
        self.cutoff_layers = cutoff_layers

        self.model = model

        if self.cutoff_layers is not None:
            if self.cutoff_layers > self.model.num_message_passing_steps:
                raise ValueError(
                    f"cutoff_layers ({self.cutoff_layers}) must be less than or equal to"
                    f" the number of message passing steps ({self.model.num_message_passing_steps})"
                )
            else:
                self.model.gnn_stacks = self.model.gnn_stacks[: self.cutoff_layers]
                self.model.num_message_passing_steps = self.cutoff_layers

        self.model_requires_grad = model_requires_grad

        if not model_requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

        warn_for_tf32_matmul()

    def predict(self, batch: base.AtomGraphs) -> Dict[str, torch.Tensor]:
        """Predict node and/or graph level attributes.

        Args:
            batch: A batch of graphs to run prediction on.

        Returns:
            A dictionary containing a node_pred tensor attribute with
            for node predictions, and a tensor of graph level predictions.
        """
        batch = self.model(batch)

        output = {}
        if self.graph_head is not None:
            output["graph_pred"] = self.graph_head.predict(batch)

        if self.stress_head is not None:
            output["stress_pred"] = self.stress_head.predict(batch)

        if self.node_head is not None:
            output["node_pred"] = self.node_head.predict(batch)

        return output

    def forward(self, batch: base.AtomGraphs) -> base.AtomGraphs:
        """Forward pass of GraphRegressor."""
        batch = self.model(batch)

        if self.graph_head is not None:
            batch = self.graph_head(batch)
        if self.stress_head is not None:
            batch = self.stress_head(batch)
        if self.node_head is not None:
            batch = self.node_head(batch)
        return batch

    def loss(self, batch: base.AtomGraphs) -> base.ModelOutput:
        """Loss function of GraphRegressor."""
        batch = self(batch)
        loss = torch.tensor(0.0)
        metrics: Dict = {}
        if self.graph_head is not None:
            graph_out = self.graph_head.loss(batch)
            metrics.update(graph_out.log)
            loss = loss.type_as(graph_out.loss)
            loss += graph_out.loss

        if self.stress_head is not None:
            stress_out = self.stress_head.loss(batch)
            metrics.update(stress_out.log)
            loss = loss.type_as(stress_out.loss)
            loss += stress_out.loss

        if self.node_head is not None:
            node_out = self.node_head.loss(batch)
            metrics.update(node_out.log)
            loss = loss.type_as(node_out.loss)
            loss += node_out.loss

        metrics["loss"] = loss.item()
        return base.ModelOutput(loss=loss, log=metrics)


def binary_accuracy(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    """Calculate binary accuracy between 2 tensors.

    Args:
        pred: the prediction tensor.
        target: the tensor of target values.
        threshold: Binary classification threshold. Default 0.5.

    Returns:
        mean accuracy.
    """
    return ((pred > threshold) == target).to(torch.float).mean().item()


def categorical_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate accuracy for K class classification.

    Args:
        pred: the tensor of logits for K classes of shape (..., K)
        target: tensor of integer target values of shape (...)

    Returns:
        mean accuracy.
    """
    pred_labels = torch.argmax(pred, dim=-1)
    return (pred_labels == target.long()).to(torch.float).mean().item()


def error_within_threshold(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.02
) -> float:
    """Calculate MAE between 2 tensors within a threshold.

    Args:
        pred: the prediction tensor.
        target: the tensor of target values.
        threshold: margin threshold. Default 0.02 (derrived from OCP metrics).

    Returns:
        Mean predictions within threshold.
    """
    error = torch.abs(pred - target)
    within_threshold = error < threshold
    return within_threshold.to(torch.float).mean().item()


def forces_within_threshold(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_num_nodes: torch.Tensor,
    threshold: float = 0.03,
) -> float:
    """Calculate MAE between batched graph tensors within a threshold.

    The predictions for a graph are counted as being within the threshold
    only if all nodes in the graph have predictions within the threshold.

    Args:
        pred: the prediction tensor.
        target: the tensor of target values.
        batch_num_nodes: A tensor containing the number of nodes per
            graph.
        threshold: margin threshold. Default 0.03 (derrived from OCP metrics).

    Returns:
        Mean predictions within threshold.
    """
    # Shape (batch_num_nodes, 3)
    error = torch.abs(pred - target)
    # Shape (batch_num_nodes)
    largest_dim_fwt = error.max(-1).values < threshold

    count_within_threshold = segment_ops.aggregate_nodes(
        largest_dim_fwt.float(), batch_num_nodes, reduction="sum"
    )
    # count equals batch_num_nodes if all nodes within threshold
    return (count_within_threshold == batch_num_nodes).to(torch.float).mean().item()


def energy_and_forces_within_threshold(
    pred_energy: torch.Tensor,
    pred_forces: torch.Tensor,
    target_energy: torch.Tensor,
    target_forces: torch.Tensor,
    batch_num_nodes: torch.Tensor,
    fixed_atoms: Optional[torch.Tensor] = None,
    threshold: Tuple[float, float] = (0.02, 0.03),
) -> float:
    """Calculate MAE between batched graph energies and forces within a threshold.

    The predictions for a graph are counted as being within the threshold
    only if all nodes in the graph have predictions within the threshold AND
    the energies are also within a threshold. A combo of the two above functions.

    Args:
        pred_*: the prediction tensors.
        target_*: the tensor of target values.
        batch_num_nodes: A tensor containing the number of nodes per
            graph.
        fixed_atoms: A tensor of bools indicating which atoms are fixed.
        threshold: margin threshold. Default (0.02, 0.03) (derrived from OCP metrics).
    Returns:
        Mean predictions within threshold.
    """
    energy_err = torch.abs(pred_energy - target_energy)
    ewt = energy_err < threshold[0]

    forces_err = torch.abs(pred_forces - target_forces)
    largest_dim_fwt = forces_err.max(-1).values < threshold[1]

    if fixed_atoms is not None:
        fixed_per_graph = segment_ops.aggregate_nodes(
            fixed_atoms.int(), batch_num_nodes, reduction="sum"
        )
        # remove the fixed atoms from the counts
        batch_num_nodes = batch_num_nodes - fixed_per_graph
        # remove the fixed atoms from the forces
        largest_dim_fwt = largest_dim_fwt[~fixed_atoms]

    force_count_within_threshold = segment_ops.aggregate_nodes(
        largest_dim_fwt.int(), batch_num_nodes, reduction="sum"
    )
    fwt = force_count_within_threshold == batch_num_nodes

    # count equals batch_num_nodes if all nodes within threshold
    return (fwt & ewt).to(torch.float).mean().item()


def _remove_fixed_atoms(
    pred_node: torch.Tensor,
    node_target: torch.Tensor,
    batch_n_node: torch.Tensor,
    fix_atoms: Optional[torch.Tensor],
    training: bool,
):
    """We use inf targets on purpose to designate nodes for removal."""
    assert len(pred_node) == len(node_target)
    if fix_atoms is not None and not training:
        pred_node = pred_node[~fix_atoms]
        node_target = node_target[~fix_atoms]
        batch_n_node = segment_ops.aggregate_nodes(
            (~fix_atoms).int(), batch_n_node, reduction="sum"
        )
    return pred_node, node_target, batch_n_node


def bce_loss(
    pred: torch.Tensor, target: torch.Tensor, metric_prefix: str = ""
) -> Tuple:
    """Binary cross-entropy loss with accuracy metric."""
    loss = torch.nn.BCEWithLogitsLoss()(pred, target.float())
    accuracy = binary_accuracy(pred, target)
    return (
        loss,
        {
            f"{metric_prefix}_accuracy": accuracy,
            f"{metric_prefix}_loss": loss.item(),
        },
    )


def cross_entropy_loss(
    pred: torch.Tensor, target: torch.Tensor, metric_prefix: str = ""
) -> Tuple:
    """Cross-entropy loss with accuracy metric."""
    loss = torch.nn.CrossEntropyLoss()(pred, target.long())
    accuracy = categorical_accuracy(pred, target)
    return (
        loss,
        {
            f"{metric_prefix}_accuracy": accuracy,
            f"{metric_prefix}_loss": loss.item(),
        },
    )

def selectively_remove_net_torque_for_nonpbc_systems(
    pred: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    n_node: torch.Tensor,
):
    """Remove net torque from non-PBC-system forces, but preserve PBC-system forces.

    Args:
        pred: The predicted forces of shape (n_atoms_in_batch, 3).
        positions: The positions of shape (n_atoms_in_batch, 3).
        cell: The cell of shape (n_batch, 3, 3).
        n_node: The number of nodes per graph, of shape (n_batch,).
    """
    nopbc_graph = torch.all(cell == 0.0, dim=(1, 2))
    if torch.any(nopbc_graph):
        if torch.all(nopbc_graph):
            pred = remove_net_torque(positions, pred, n_node)
        else:
            # Handle a mixed batch of pbc and non-pbc systems
            batch_indices = torch.repeat_interleave(
                torch.arange(cell.size(0), device=n_node.device), n_node
            )
            nopbc_atom = nopbc_graph[batch_indices]
            adjusted_pred_non_pbc = remove_net_torque(
                positions[nopbc_atom], pred[nopbc_atom], n_node[nopbc_graph]
            )
            pred = pred.clone()
            pred[nopbc_atom] = adjusted_pred_non_pbc

    return pred


def remove_net_torque(
    positions: torch.Tensor,
    forces: torch.Tensor,
    n_nodes: torch.Tensor,
) -> torch.Tensor:
    """Adjust the predicted forces to eliminate net torque for each graph in the batch.

    We frame the problem of net-torque-elimination as a constrained optimisation problem;
    what is the minimal additive adjustment (in L2 norm) that eliminates net torque?

    This analytically solvable with Lagrange multipliers and the solution involves cheap
    linear algebra operations (cross products and the inversion of 3x3 matrices).

    Args:
        positions : torch.Tensor of shape (N, 3)
            Positions of atoms (concatenated for all graphs in the batch).
        forces : torch.Tensor of shape (N, 3)
            Predicted forces on atoms.
        n_nodes : torch.Tensor of shape (B,)
            Number of nodes in each graph, where B is the number of graphs in the batch.

    Returns:
        adjusted_forces : torch.Tensor of shape (N, 3)
            Adjusted forces with zero net torque and net force for each graph.
    """
    B = n_nodes.shape[0]
    tau_total, r = compute_net_torque(positions, forces, n_nodes)

    # Compute scalar s per graph: sum_i ||r_i||^2
    r_squared = torch.sum(r**2, dim=1)  # Shape: (N,)
    s = segment_ops.aggregate_nodes(r_squared, n_nodes, "sum")  # Shape: (B,)

    # Compute matrix S per graph: sum_i outer(r_i, r_i)
    r_unsqueezed = r.unsqueeze(2)  # Shape: (N, 3, 1)
    r_T_unsqueezed = r.unsqueeze(1)  # Shape: (N, 1, 3)
    outer_products = r_unsqueezed @ r_T_unsqueezed  # Shape: (N, 3, 3)
    S = segment_ops.aggregate_nodes(outer_products, n_nodes, "sum")  # Shape: (B, 3, 3)

    # Compute M = S - sI
    I = (  # noqa: E741
        torch.eye(3, device=positions.device).unsqueeze(0).expand(B, -1, -1)
    )  # Shape: (B, 3, 3)
    M = S - (s.view(-1, 1, 1)) * I  # Shape: (B, 3, 3)

    # Right-hand side vector b per graph
    b = -tau_total  # Shape: (B, 3)

    # Solve M * mu = b for mu per graph
    try:
        mu = torch.linalg.solve(M, b.unsqueeze(2)).squeeze(2)  # Shape: (B, 3)
    except RuntimeError:
        # Handle singular matrix M by using the pseudo-inverse
        M_pinv = torch.linalg.pinv(M)  # Shape: (B, 3, 3)
        mu = torch.bmm(M_pinv, b.unsqueeze(2)).squeeze(2)  # Shape: (B, 3)

    # Compute adjustments to forces
    mu_batch = torch.repeat_interleave(mu, n_nodes, dim=0)  # Shape: (N, 3)
    forces_delta = torch.linalg.cross(r, mu_batch)  # Shape: (N, 3)

    # Adjusted forces
    adjusted_forces = forces + forces_delta  # Shape: (N, 3)

    return adjusted_forces


def compute_net_torque(
    positions: torch.Tensor,
    forces: torch.Tensor,
    n_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the net torque on a system of particles."""
    com = segment_ops.aggregate_nodes(positions, n_nodes, "mean")
    com_repeat = torch.repeat_interleave(com, n_nodes, dim=0)  # Shape: (N, 3)
    com_relative_positions = positions - com_repeat  # Shape: (N, 3)
    torques = torch.linalg.cross(com_relative_positions, forces)  # Shape: (N, 3)
    net_torque = segment_ops.aggregate_nodes(torques, n_nodes, "sum")
    return net_torque, com_relative_positions
