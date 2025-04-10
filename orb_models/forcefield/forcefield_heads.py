import numpy
import torch
from typing import Literal, Optional, Union

from orb_models.forcefield.loss import forces_loss_function
from orb_models.forcefield.property_definitions import PROPERTIES, PropertyDefinition
from orb_models.forcefield.forcefield_utils import (
    conditional_huber_force_loss,
    forces_within_threshold,
    remove_fixed_atoms,
    maybe_remove_net_force_and_torque,
)
from orb_models.forcefield.reference_energies import REFERENCE_ENERGIES
from orb_models.forcefield import base, segment_ops
from orb_models.forcefield.forcefield_utils import bucketed_mean_error, mean_error
from orb_models.forcefield.nn_util import build_mlp, ScalarNormalizer


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
            self.linear.weight.data = torch.tensor(
                weight_init, dtype=torch.get_default_dtype()
            )
        if not trainable:
            self.linear.weight.requires_grad = False

    def forward(self, atom_types: torch.Tensor, n_node: torch.Tensor) -> torch.Tensor:
        """Forward pass of the LinearReferenceEnergy.

        Args:
            atom_types: A tensor of atomic numbers of shape (n_atoms,)

        Returns:
            A tensor of shape (n_graphs,) containing the reference energy.
        """
        # NOTE: we explictly make the one-hot tensor here to avoid the
        # alternative k-hot embedding used atom type diffusion models.
        one_hot_atomic = torch.nn.functional.one_hot(atom_types, num_classes=118).to(
            self.linear.weight.dtype
        )
        reduced = segment_ops.aggregate_nodes(one_hot_atomic, n_node, reduction="sum")
        return self.linear(reduced).squeeze(-1)


class EnergyHead(torch.nn.Module):
    """Energy prediction head that can be appended to a base model."""

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        level_of_theory: Optional[str] = None,
        predict_atom_avg: bool = True,
        loss_type: Literal["mae", "mse", "huber_0.01"] = "huber_0.01",
        dropout: Optional[float] = None,
        checkpoint: Optional[str] = None,
        online_normalisation: bool = True,
        activation: str = "ssp",
        reference_energy: Optional[str] = None,
    ):
        """Initializes the EnergyHead MLP.

        Args:
            latent_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            level_of_theory: The method used to compute the gold energies e.g. "SCAN"/ "D3" / "D4".
                If provided, PROPERTIES['forces-{level_of_theory}'] will be used.
            predict_atom_avg: Whether to predict the average atom energy or total.
            loss_type: The type of loss to use. Either "mae", "mse" or "huber_x"
                where x is the delta parameter for the huber loss.
            dropout: The level of dropout to apply.
            checkpoint: Whether to use checkpointing to save memory at the cost of extra compute.
                None (no checkpointing), 'reentrant' or 'non-reentrant'.
            online_normalisation: Whether to update the normalisation statistics online.
            activation: The activation function to use.
            reference_energy: The reference energy to use. If None, use the default reference energy.
        """
        super().__init__()
        self.target_name = f"energy-{level_of_theory}" if level_of_theory else "energy"
        self.target = PROPERTIES[self.target_name]

        ref_energy_name = reference_energy or (
            f"{level_of_theory}-shifted" if level_of_theory else "vasp-shifted"
        )
        ref = REFERENCE_ENERGIES[ref_energy_name]
        if predict_atom_avg:
            self.node_aggregation = "mean"
            means = torch.tensor(
                [ref.residual_mean_per_atom],
            )
            stds = torch.tensor([ref.residual_std_per_atom])
        else:
            self.node_aggregation = "sum"
            means = torch.tensor([ref.residual_mean])
            stds = torch.tensor([ref.residual_std])

        self.normalizer = ScalarNormalizer(
            init_mean=means, init_std=stds, online=online_normalisation
        )
        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=1,
            activation=activation,
            dropout=dropout,
            checkpoint=checkpoint,
        )
        self.reference = LinearReferenceEnergy(
            weight_init=ref.coefficients, trainable=False
        )
        self.atom_avg = predict_atom_avg
        self.loss_type = loss_type

    def forward(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Forward pass (without inverse transformation)."""
        input = segment_ops.aggregate_nodes(
            node_features, batch.n_node, reduction=self.node_aggregation
        )
        pred = self.mlp(input)
        return pred.squeeze(-1)

    def predict(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Predict energy."""
        pred = self(node_features, batch)
        return self.denormalize(pred, batch)

    def loss(
        self,
        pred: torch.Tensor,
        batch: base.AtomGraphs,
        alternative_target: Optional[torch.Tensor] = None,
    ):
        """Apply mlp to compute loss and metrics."""
        name = self.target.fullname
        pred = pred.reshape(-1)
        if alternative_target is not None:
            raw_target = alternative_target.reshape(-1)
        else:
            raw_target = batch.system_targets[name].reshape(-1)

        reference = self.reference(batch.atomic_numbers, batch.n_node).reshape(-1)
        target = self.normalize(raw_target, batch, reference)
        assert (
            pred.shape == raw_target.shape == target.shape
        ), f"{pred.shape} != {raw_target.shape} != {target.shape}"

        loss = mean_error(pred, target, self.loss_type)

        reference_error = raw_target - reference
        if self.atom_avg:
            reference_error = reference_error / batch.n_node
        raw_pred = self.denormalize(pred, batch)
        metrics = {
            f"{name}_loss": loss,
            f"{name}_mae_raw": torch.abs(raw_pred - raw_target).mean(),
            f"{name}_mse_raw": ((raw_pred - raw_target) ** 2).mean(),
            f"{name}_reference_mae": torch.abs(reference_error).mean(),
        }
        return base.ModelOutput(loss=loss, log=metrics)

    def denormalize(self, x: torch.Tensor, batch: base.AtomGraphs):
        """Denormalize the energy prediction."""
        x = self.normalizer.inverse(x).squeeze(-1)
        if self.atom_avg:
            x = x * batch.n_node
        return x + self.reference(batch.atomic_numbers, batch.n_node)

    def normalize(
        self,
        x: torch.Tensor,
        batch: base.AtomGraphs,
        reference: Optional[torch.Tensor] = None,
        online: Optional[bool] = None,
    ):
        """Normalize the energy prediction."""
        if reference is None:
            reference = self.reference(batch.atomic_numbers, batch.n_node)
        x = x - reference
        if self.atom_avg:
            x = x / batch.n_node
        return self.normalizer(x, online=online)


class ForceHead(torch.nn.Module):
    """Force prediction head that can be appended to a base model."""

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        level_of_theory: Optional[str] = None,
        remove_mean: bool = True,
        remove_torque_for_nonpbc_systems: bool = True,
        loss_type: Literal[
            "mae", "mse", "huber_0.01", "condhuber_0.01"
        ] = "condhuber_0.01",
        dropout: Optional[float] = None,
        checkpoint: Optional[str] = None,
        output_size: int = 3,
        online_normalisation: bool = True,
        activation: str = "ssp",
        detach_node_features: bool = False,
    ):
        """Initializes the ForceHead MLP.

        Args:
            latent_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            level_of_theory: The method used to compute the gold energies e.g. "SCAN"/ "D3" / "D4".
                If provided, PROPERTIES['forces-{level_of_theory}'] will be used.
            remove_mean: Whether to remove the mean of the predictions.
            remove_torque_for_nonpbc_systems: Whether to remove the net torque of force predictions
                for systems without periodic boundary conditions.
            loss_type: The type of loss to use. Either "mae" or "mse".
            dropout: The level of dropout to apply.
            checkpoint: Whether to use checkpointing to save memory at the cost of extra compute.
                None (no checkpointing), 'reentrant' or 'non-reentrant'.
            output_size: The size of the output layer.
            online_normalisation: Whether to update the normalisation statistics online.
            activation: The activation function to use.
            detach_node_features: If True, detaches node features from computational graph.
                This means that the force loss has no impact on training the underlying
                forcefield model.
        """
        super().__init__()
        target_name = f"forces-{level_of_theory}" if level_of_theory else "forces"
        self.target = PROPERTIES[target_name]
        self.normalizer = ScalarNormalizer(online=online_normalisation)
        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=output_size,
            activation=activation,
            dropout=dropout,
            checkpoint=checkpoint,
        )
        assert isinstance(remove_mean, bool)
        assert isinstance(remove_torque_for_nonpbc_systems, bool)
        self.remove_mean = remove_mean
        self.remove_torque_for_nonpbc_systems = remove_torque_for_nonpbc_systems
        self.loss_type = loss_type
        self.detach_node_features = detach_node_features

    def forward(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Forward pass (without inverse normalisation)."""
        if self.detach_node_features:
            node_features = node_features.detach()
        pred = self.mlp(node_features)
        pred = maybe_remove_net_force_and_torque(
            batch, pred, self.remove_mean, self.remove_torque_for_nonpbc_systems
        )
        return pred

    def predict(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Predict forces."""
        pred = self(node_features, batch)
        return self.normalizer.inverse(pred)

    def loss(
        self,
        pred: torch.Tensor,
        batch: base.AtomGraphs,
        alternative_target: Optional[torch.Tensor] = None,
    ):
        """Compute loss and metrics."""
        name = self.target.fullname
        gold_target = batch.node_targets[name]
        raw_target = gold_target if alternative_target is None else alternative_target
        return forces_loss_function(
            pred=pred,
            raw_target=raw_target,
            raw_gold_target=gold_target,
            name=name,
            normalizer=self.normalizer,
            n_node=batch.n_node,
            fix_atoms=batch.fix_atoms,
            loss_type=self.loss_type,
            training=self.training,
        )

    def denormalize(self, x: torch.Tensor, batch: base.AtomGraphs):
        """Denormalize the force prediction."""
        return self.normalizer.inverse(x)

    def normalize(
        self, x: torch.Tensor, batch: base.AtomGraphs, online: Optional[bool] = None
    ):
        """Normalize the force prediction."""
        return self.normalizer(x, online=online)


class StressHead(torch.nn.Module):
    """MLP Regression head for stress."""

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        level_of_theory: Optional[str] = None,
        node_aggregation: Literal["sum", "mean"] = "mean",
        loss_type: Literal["mae", "mse", "huber_0.01"] = "huber_0.01",
        dropout: Optional[float] = None,
        checkpoint: Optional[str] = None,
        online_normalisation: bool = True,
        activation: str = "ssp",
        off_diag_loss_weight: float = 0.1,
    ):
        """Initializes the StressHead MLP.

        Args:
            latent_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            level_of_theory: The method used to compute the gold energies e.g. "SCAN"/ "D3" / "D4".
                If provided, PROPERTIES['forces-{level_of_theory}'] will be used.
            node_aggregation: The method for aggregating the node features
                from the pretrained model representations.
            loss_type: The type of loss to use. Either "mae", "mse", or "huber_x"
                where x is the delta parameter for the huber loss.
            dropout: The level of dropout to apply.
            checkpoint: Whether to use PyTorch checkpointing.
                None (no checkpointing), 'reentrant' or 'non-reentrant'.
            online_normalisation: Whether to normalise the target online.
            activation: The activation function to use.
            off_diag_loss_weight: The weight of the off-diagonal stress loss.
        """
        super().__init__()
        target_name = f"stress-{level_of_theory}" if level_of_theory else "stress"
        self.target = PROPERTIES[target_name]
        self.off_diag_loss_weight = off_diag_loss_weight

        self.diag_normalizer = ScalarNormalizer(online=online_normalisation)
        self.offdiag_normalizer = ScalarNormalizer(online=online_normalisation)
        self.node_aggregation = node_aggregation
        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=self.target.dim,
            activation=activation,
            dropout=dropout,
            checkpoint=checkpoint,
        )
        self.loss_type = loss_type

    def forward(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Predictions with raw logits (no sigmoid/softmax or any inverse transformations)."""
        input = segment_ops.aggregate_nodes(
            node_features,
            batch.n_node,
            reduction=self.node_aggregation,
        )
        pred = self.mlp(input)
        return pred

    def predict(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Predict stress in eV/Å^3."""
        pred = self(node_features, batch)
        return self.denormalize(pred, batch)

    def loss(
        self,
        pred: torch.Tensor,
        batch: base.AtomGraphs,
        alternative_target: Optional[torch.Tensor] = None,
    ):
        """Apply mlp to compute loss and metrics."""
        name = self.target.fullname
        if alternative_target is not None:
            target = alternative_target
        else:
            target = batch.system_targets[name]

        assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"

        normalized_target = self.normalize(target, batch)
        loss_diag = mean_error(pred[:, :3], normalized_target[:, :3], self.loss_type)
        loss_offdiag = mean_error(pred[:, 3:], normalized_target[:, 3:], self.loss_type)
        loss = loss_diag + (self.off_diag_loss_weight * loss_offdiag)

        raw_pred = self.denormalize(pred, batch)
        metrics = {
            f"{name}_loss": loss,
            f"{name}_mae_raw": torch.abs(raw_pred - target).mean(),
            f"{name}_mse_raw": ((raw_pred - target) ** 2).mean(),
            f"{name}_diag_mae_raw": torch.abs(raw_pred[:, :3] - target[:, :3]).mean(),
            f"{name}_diag_mse_raw": ((raw_pred[:, :3] - target[:, :3]) ** 2).mean(),
            f"{name}_offdiag_mae_raw": torch.abs(
                raw_pred[:, 3:] - target[:, 3:]
            ).mean(),
            f"{name}_offdiag_mse_raw": ((raw_pred[:, 3:] - target[:, 3:]) ** 2).mean(),
        }

        return base.ModelOutput(loss=loss, log=metrics)

    def denormalize(self, pred: torch.Tensor, batch: base.AtomGraphs):
        """Denormalize the stress prediction."""
        diag = self.diag_normalizer.inverse(pred[:, :3])
        offdiag = self.offdiag_normalizer.inverse(pred[:, 3:])
        out = torch.cat([diag, offdiag], dim=-1)
        return out

    def normalize(
        self, x: torch.Tensor, batch: base.AtomGraphs, online: Optional[bool] = None
    ):
        """Normalize the stress prediction."""
        diag = self.diag_normalizer(x[:, :3], online=online)
        offdiag = self.offdiag_normalizer(x[:, 3:], online=online)
        out = torch.cat([diag, offdiag], dim=-1)
        return out


def confidence_row_fn(row, dataset_name: str):
    """Stub function for confidence property definition."""
    raise NotImplementedError(
        "Confidence is intrinsically defined, and not a property."
    )


_confidence = PropertyDefinition(
    name="confidence",
    dim=3,
    domain="real",
    row_to_property_fn=confidence_row_fn,
)


class ConfidenceHead(torch.nn.Module):
    """Confidence prediction head that estimates mean force prediction error.

    This module produces an intrinsic force estimate of the mean error of the
    3-dimensional forces for a single atom.

    """

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        num_bins: int = 50,
        max_error: float = 0.3,
        binning_scale: str = "linear",
        dropout: Optional[float] = None,
        activation: str = "ssp",
        detach_node_features: bool = True,
        hard_clamp: bool = True,
    ):
        """Initializes the ConfidenceHead MLP.

        Args:
            latent_dim: Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers: Number of MLP layers.
            mlp_hidden_dim: MLP hidden size.
            num_bins: Number of bins to divide the error range into.
            max_error: Maximum mean error value (in eV/Å) to consider for binning.
            binning_scale: The scale for the bins, either linear or exponential.
            dropout: The level of dropout to apply.
            activation: The activation function to use.
            detach_node_features: If True, detaches node features from computational graph.
                This means that the confidence loss has no impact on training the underlying
                forcefield model.
            hard_clamp: If True, ignore any errors above max_error such that they do not contribute
                to the loss, rather than just clamping them to the max_bin.
        """
        super().__init__()
        self.target = _confidence
        self.num_bins = num_bins
        self.max_error = max_error
        self.detach_node_features = detach_node_features
        self.hard_clamp = hard_clamp
        self.ignore_index = -100
        if binning_scale == "linear":
            bins = torch.linspace(0.0, max_error, int(num_bins + 1))
        elif binning_scale == "exponential":
            bins = torch.linspace(
                numpy.log(1e-18), numpy.log(max_error), int(num_bins + 1)
            ).exp()
            # Can't use zero above, but we always want the scale to start
            # at zero, not just "really small".
            bins[0] = 0.0
        else:
            raise ValueError("Invalid binning scale. Use exponential or linear.")

        self.register_buffer("bin_edges", bins)

        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=num_bins,
            activation=activation,
            dropout=dropout,
        )

    def get_error_bins(self, force_error: torch.Tensor) -> torch.Tensor:
        """Convert force errors to bin indices.

        Args:
            force_error: Force error magnitudes of shape (n_atoms,)

        Returns:
            Bin indices of shape (n_atoms,)
        """
        clamped_error = torch.clamp(force_error, 0, self.max_error)
        bins = torch.bucketize(clamped_error, self.bin_edges) - 1  # type: ignore
        clamped = torch.clamp(bins, 0, self.num_bins - 1)

        if self.hard_clamp:
            clamped[force_error > self.max_error] = self.ignore_index
        return clamped

    def forward(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Forward pass to predict error bin probabilities.

        Returns:
            Dictionary with bin probabilities of shape (n_atoms, num_bins)
        """
        if self.detach_node_features:
            node_features = node_features.detach()
        logits = self.mlp(node_features)
        return logits

    def predict(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Predict bin probabilities."""
        logits = self(node_features, batch)
        return torch.softmax(logits, dim=-1)

    def loss(
        self,
        confidence_logits: torch.Tensor,
        force_error: torch.Tensor,
        batch: base.AtomGraphs,
    ) -> base.ModelOutput:
        """Compute loss and metrics for confidence prediction.

        Args:
            confidence_logits: Predicted bin logits from forward() of shape (n_atoms, num_bins)
            force_error: True force error magnitudes of shape (n_atoms,)
            batch: Graph batch information
        """
        dtype = confidence_logits.dtype
        # Get true bin indices
        true_bins = self.get_error_bins(force_error)

        # Cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            confidence_logits, true_bins, ignore_index=self.ignore_index
        )

        # Calculate accuracy
        pred_bins = torch.argmax(confidence_logits, dim=-1)
        accuracy = (pred_bins == true_bins).to(dtype).mean()

        metrics = {
            "confidence_loss": loss,
            "confidence_accuracy": accuracy,
        }

        return base.ModelOutput(loss=loss, log=metrics)


class GraphHead(torch.nn.Module):
    """MLP Regression head that can be appended to a base model.

    NOTE: This head is deprecated. Use EnergyHead, ForceHead, or StressHead instead.
    We only support this head for backwards compatibility with orb-v2 models.
    """

    def __init__(
        self,
        latent_dim: int,
        num_mlp_layers: int,
        mlp_hidden_dim: int,
        target: Union[str, PropertyDefinition],
        node_aggregation: Literal["sum", "mean"] = "mean",
        real_loss_type: Literal["mae", "mse", "huber_0.01"] = "huber_0.01",
        dropout: Optional[float] = None,
        checkpoint: Optional[str] = None,
        online_normalisation: bool = True,
        activation: str = "ssp",
    ):
        """Initializes the GraphHead MLP.

        Args:
            latent_dim (int): Dimensionality of the incoming latent vector from the base model.
            num_mlp_layers (int): Number of MLP layers.
            mlp_hidden_dim (int): MLP hidden size.
            target: either the name of a PropertyDefinition or a PropertyDefinition itself
            node_aggregation: The method for aggregating the node features
                from the pretrained model representations.
            loss_type: The type of loss to use. Either "mae", "mse", or "huber_x"
                where x is the delta parameter for the huber loss.
            dropout: The level of dropout to apply.
            checkpoint: Whether to use PyTorch checkpointing.
                    None (no checkpointing), 'reentrant' or 'non-reentrant'.
            online_normalisation: Whether to normalize the target online.
            activation: The activation function to use.
        """
        super().__init__()
        self.target = PROPERTIES[target] if isinstance(target, str) else target

        self.normalizer = ScalarNormalizer(
            init_mean=self.target.means,
            init_std=self.target.stds,
            online=online_normalisation,
        )
        self.node_aggregation = node_aggregation
        self.mlp = build_mlp(
            input_size=latent_dim,
            hidden_layer_sizes=[mlp_hidden_dim] * num_mlp_layers,
            output_size=self.target.dim,
            activation=activation,
            dropout=dropout,
            checkpoint=checkpoint,
        )
        activation_dict = {
            "real": torch.nn.Identity,
            "binary": torch.nn.Sigmoid,
            "categorical": torch.nn.Softmax,
        }
        self.output_activation = activation_dict[self.target.domain]()
        self.real_loss_type = real_loss_type

    def forward(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Predictions with raw logits (no sigmoid/softmax or any inverse transformations)."""
        input = segment_ops.aggregate_nodes(
            node_features,
            batch.n_node,
            reduction=self.node_aggregation,
        )
        pred = self.mlp(input)
        return pred

    def predict(
        self, node_features: torch.Tensor, batch: base.AtomGraphs
    ) -> torch.Tensor:
        """Predict graph-level attribute."""
        pred = self(node_features, batch)
        logits = pred.squeeze(-1)
        probs = self.output_activation(logits)
        if self.target.domain == "real":
            probs = self.normalizer.inverse(probs)
        return probs

    def loss(
        self,
        pred: torch.Tensor,
        batch: base.AtomGraphs,
        alternative_target: Optional[torch.Tensor] = None,
    ):
        """Apply mlp to compute loss and metrics.

        Depending on whether the target is real/binary/categorical, we
        use an MSE/cross-entropy loss. In the case of cross-entropy, the
        preds are logits (not normalized) to take advantage of numerically
        stable log-softmax.
        """
        name = self.target.fullname
        if alternative_target is not None:
            target = alternative_target
        else:
            target = batch.system_targets[name].squeeze(-1)
        pred = pred.squeeze(-1)

        assert pred.shape == target.shape, f"{pred.shape} != {target.shape}"
        normalized_target = self.normalizer(target)
        loss = mean_error(pred, normalized_target, self.real_loss_type)
        raw_pred = self.normalizer.inverse(pred)
        metrics = {
            f"{name}_loss": loss,
            f"{name}_mae_raw": torch.abs(raw_pred - target).mean(),
            f"{name}_mse_raw": ((raw_pred - target) ** 2).mean(),
        }

        return base.ModelOutput(loss=loss, log=metrics)
