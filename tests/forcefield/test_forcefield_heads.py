import numpy as np
import pytest
import torch
from ase.stress import full_3x3_to_voigt_6_stress

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.models.nn_util import ScalarNormalizer
from orb_models.forcefield.models.forcefield_heads import (
    ConfidenceHead,
    ForceHead,
    LinearReferenceEnergy,
)
from orb_models.forcefield.models.forcefield_utils import torch_full_3x3_to_voigt_6_stress
from tests.forcefield.conftest import _KEY, get_batch_from_ase_with_latents


@pytest.mark.parametrize(
    "weight_init, trainable",
    [
        (None, None),
        (None, True),
        (None, False),
        (np.random.rand(1, 118).astype(np.float32), None),
        (np.random.rand(1, 118).astype(np.float32), True),
        (np.random.rand(1, 118).astype(np.float32), False),
    ],
)
def test_linear_reference_energy_initialization(weight_init, trainable):
    model = LinearReferenceEnergy(weight_init=weight_init, trainable=trainable)
    if weight_init is not None:
        assert torch.allclose(model.linear.weight.data, torch.tensor(weight_init))
    expected_trainable = trainable if trainable is not None else weight_init is None
    assert model.linear.weight.requires_grad == expected_trainable


def test_linear_reference_energy_forward_zero_weights():
    weight_init = np.zeros((1, 118), dtype=np.float32)
    model = LinearReferenceEnergy(weight_init=weight_init, trainable=False)
    atom_types = torch.tensor([1, 6, 8, 1, 1], dtype=torch.long)
    n_node = torch.tensor([5])
    output = model.forward(atom_types, n_node)
    expected_output = torch.zeros((1, 1))
    assert torch.allclose(output, expected_output)


def test_linear_reference_energy_forward_one_weights():
    weight_init = np.ones((1, 118), dtype=np.float32)
    model = LinearReferenceEnergy(weight_init=weight_init, trainable=False)
    atom_types = torch.tensor([1, 6, 8, 1, 1], dtype=torch.long)
    n_node = torch.tensor([5])
    output = model.forward(atom_types, n_node)
    expected_output = torch.tensor([[5.0]])
    assert torch.allclose(output, expected_output)


def test_energy_head_initialization(energy_head):
    assert energy_head.target.fullname == "energy"
    assert isinstance(energy_head.mlp, torch.nn.Module)
    assert isinstance(energy_head.normalizer, ScalarNormalizer)
    assert isinstance(energy_head.reference, LinearReferenceEnergy)
    assert energy_head.node_aggregation == "mean"


def test_energy_head_forward(energy_head, batch):
    node_features = batch.node_features[_KEY]
    output = energy_head.forward(node_features, batch)
    assert output.shape == (batch.n_node.shape[0], 1)


def test_energy_head_can_predict(energy_head, batch):
    energy_pred = energy_head.predict(batch.node_features[_KEY], batch)
    pred1, pred2 = energy_pred.chunk(2)

    assert pred1.shape == pred2.shape == (1,)
    assert torch.isfinite(energy_pred).all()
    assert torch.allclose(pred1, pred2)


def test_energy_head_loss(energy_head, batch):
    node_features = batch.node_features[_KEY]
    output = energy_head(node_features, batch)
    model_output = energy_head.loss(output, batch)
    assert model_output.loss.shape == ()
    assert torch.isfinite(model_output.loss)
    assert "energy_loss" in model_output.log
    assert "energy_mae_raw" in model_output.log


def test_force_head_initialization(force_head):
    assert force_head.target.fullname == "forces"
    assert force_head.loss_type == "mae"
    assert isinstance(force_head.mlp, torch.nn.Module)
    assert isinstance(force_head.normalizer, ScalarNormalizer)
    assert force_head.normalizer.online
    assert force_head.remove_mean
    assert force_head.remove_torque_for_nonpbc_systems


def test_force_head_forward(force_head, batch):
    node_features = batch.node_features[_KEY]
    output = force_head.forward(node_features, batch)
    assert output.shape == (batch.n_node.sum().item(), 3)


def test_force_head_predict(force_head, batch):
    natoms = batch.n_node[0]
    node_features = batch.node_features[_KEY]
    forces_pred = force_head.predict(node_features, batch)
    pred1, pred2 = forces_pred.chunk(2)

    assert pred1.shape == pred2.shape == (natoms, 3)
    assert torch.isfinite(pred1).all()
    assert torch.allclose(pred1, pred2)


@pytest.mark.parametrize("loss_type", ["mae", "mse"])
def test_force_head_loss(loss_type, batch):
    force_head = ForceHead(
        latent_dim=batch.node_features[_KEY].shape[1],
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        loss_type=loss_type,
    )
    batch = AtomGraphs.batch([batch, batch, batch])
    node_features = batch.node_features[_KEY]
    output = force_head(node_features, batch)
    model_output = force_head.loss(output, batch)
    loss, log = model_output.loss, model_output.log
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert "forces_loss" in log
    assert f"forces_{loss_type}_raw" in log
    assert "forces_cosine_sim" in log
    assert "forces_wt_0.03" in log


def test_force_head_detach_node_features(batch):
    force_head = ForceHead(
        latent_dim=batch.node_features[_KEY].shape[1],
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        loss_type="mae",
        detach_node_features=True,
    )

    node_features = batch.node_features[_KEY]
    pre_linear = torch.nn.Linear(node_features.shape[1], node_features.shape[1])
    node_features = pre_linear(node_features)

    output = force_head(node_features, batch)
    loss_output = force_head.loss(output, batch)
    loss, log = loss_output.loss, loss_output.log
    assert torch.isfinite(loss)
    assert loss.shape == ()
    assert "forces_loss" in log
    assert "forces_mae_raw" in log

    assert pre_linear.weight.grad is None


def test_confidence_head_binning():
    confidence_head = ConfidenceHead(
        latent_dim=3,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        binning_scale="linear",
        num_bins=13,
        max_error=26.0,
        hard_clamp=False,
    )
    errors = torch.tensor([1.23, 0.09, 3.2, 18.0])
    bins = confidence_head.get_error_bins(errors)
    assert torch.allclose(bins, torch.tensor([0, 0, 1, 8]))

    # OOD error should be max possible value
    bins = confidence_head.get_error_bins(torch.tensor([100.0]))
    assert bins.item() == 12.0

    # With hard clamp, OOD error should be ignored
    confidence_head = ConfidenceHead(
        latent_dim=3,
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        binning_scale="linear",
        num_bins=13,
        max_error=26.0,
        hard_clamp=True,
    )
    errors = torch.tensor([1.23, 0.09, 3.2, 18.0])
    bins = confidence_head.get_error_bins(errors)
    assert torch.allclose(bins, torch.tensor([0, 0, 1, 8]))

    bins = confidence_head.get_error_bins(torch.tensor([100.0]))
    assert bins.item() == -100


def test_confidence_head_forward_and_loss(batch):
    force_head = ForceHead(
        latent_dim=batch.node_features[_KEY].shape[1],
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        loss_type="mae",
    )

    confidence_head = ConfidenceHead(
        latent_dim=batch.node_features[_KEY].shape[1],
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        hard_clamp=False,
    )
    batch = AtomGraphs.batch([batch, batch])
    node_features = batch.node_features[_KEY]
    forces_out = force_head(node_features, batch)
    forces_target = batch.node_targets["forces"]
    force_err = torch.abs(forces_out - forces_target).mean(dim=-1)

    logits = confidence_head(node_features, batch)

    loss_output = confidence_head.loss(logits, force_err, batch)
    loss, log = loss_output.loss, loss_output.log

    batch_n_node = batch.n_node.sum().item()
    assert logits.shape == (batch.n_node.sum().item(), confidence_head.num_bins)
    assert torch.allclose(torch.softmax(logits, dim=-1).sum(-1), torch.ones(batch_n_node))

    assert torch.isfinite(loss)
    assert loss.shape == ()

    assert "confidence_loss" in log
    assert "confidence_accuracy" in log


def test_confidence_head_detach_node_features(batch):
    force_head = ForceHead(
        latent_dim=batch.node_features[_KEY].shape[1],
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        loss_type="mae",
    )
    confidence_head = ConfidenceHead(
        latent_dim=batch.node_features[_KEY].shape[1],
        num_mlp_layers=1,
        mlp_hidden_dim=16,
        detach_node_features=True,
        hard_clamp=False,
    )

    node_features = batch.node_features[_KEY]
    forces_out = force_head(node_features, batch)
    forces_target = batch.node_targets["forces"]
    force_err = torch.abs(forces_out - forces_target).mean(dim=-1)

    # if we add a linear layer before the mlp, if the node features are detached,
    # the gradient should not flow through the linear layer
    pre_linear = torch.nn.Linear(node_features.shape[1], node_features.shape[1])
    logits = confidence_head(pre_linear(node_features), batch)

    print("logits", logits)
    loss_output = confidence_head.loss(logits, force_err, batch)
    loss, log = loss_output.loss, loss_output.log
    loss.backward()
    print("loss", loss)
    assert torch.isfinite(loss)
    assert loss.shape == ()

    for name, param in confidence_head.named_parameters():
        assert param.grad is not None
    assert pre_linear.weight.grad is None

    assert "confidence_loss" in log
    assert "confidence_accuracy" in log


@pytest.mark.parametrize("head_name", ["energy_head", "force_head"])
def test_energy_and_force_train_and_eval(request, head_name):
    """A mini-integration test for train+eval of energy+force heads.

    Specifically, we check that:
        - Training runs without error and changes all params and buffers.
        - Once in eval mode post-training, the model is deterministic.
    """
    head = request.getfixturevalue(head_name)
    orig_head_params = {k: v.clone().detach() for k, v in head.named_parameters()}

    batch_list = [get_batch_from_ase_with_latents() for _ in range(10)]
    batch = AtomGraphs.batch(batch_list)
    natoms = batch_list[0].n_node[0]

    if head_name == "energy_head":
        batch.system_targets["energy"] = head.reference(
            batch.atomic_numbers, batch.n_node
        ).unsqueeze(-1) + (batch.system_targets["energy"] * natoms)

    # Simulate training
    optim = torch.optim.Adam(head.parameters(), lr=1e-2)
    head.train()
    for _ in range(10):
        optim.zero_grad()
        node_features = batch.node_features[_KEY]
        output = head(node_features, batch)
        out = head.loss(output, batch)
        assert torch.isfinite(out.loss).all()
        out.loss.sum().backward()
        optim.step()

    # Check params have changed (except linear reference)
    for name, param in head.named_parameters():
        if "bias" in name and param.dim() == 1:
            # Don't check scalar biases, they often don't change
            continue
        if name == "reference.linear.weight":
            assert torch.allclose(param, orig_head_params[name])
        else:
            assert not torch.allclose(param, orig_head_params[name]), name

    # Check that head predictions are deterministic once we are in eval mode
    head.eval()
    node_features = batch.node_features[_KEY]
    pred_after = head.predict(node_features, batch)
    pred_after2 = head.predict(node_features, batch)
    assert torch.allclose(pred_after, pred_after2)


@pytest.mark.parametrize("online_normalisation", [False, True])
@pytest.mark.parametrize("head_name", ["energy_head", "force_head"])
def test_energy_and_force_normalization_in_train_loop(request, head_name, online_normalisation):
    """Integration-like test for learning energy/forces that require normalisation due to large std."""
    head = request.getfixturevalue(head_name)

    # reinstantiate head with online_normalisation and no initial mean or std
    head.normalizer = ScalarNormalizer(online=online_normalisation)
    orig_head_buffers = {k: v.clone().detach() for k, v in head.named_buffers()}

    batch_list = [get_batch_from_ase_with_latents() for _ in range(10)]
    batch = AtomGraphs.batch(batch_list)
    natoms = batch_list[0].n_node[0]

    # Increase the scale of the targets by 10x.
    # Note that, for energies, the model's target is the diff
    # from the reference energy (divided by natoms)
    if head_name == "energy_head":
        name = "energy"
        batch.system_targets[name] = head.reference(batch.atomic_numbers, batch.n_node).unsqueeze(
            -1
        ) + (batch.system_targets[name] * 10 * natoms)
    else:
        name = "forces"
        batch.node_targets[name] *= 10

    # Predictions before training
    node_features = batch.node_features[_KEY]
    pred_before = head.predict(node_features, batch)

    # Simulate training
    optim = torch.optim.Adam(head.parameters(), lr=1e-4)
    head.train()
    for _ in range(5):
        optim.zero_grad()
        node_features = batch.node_features[_KEY]
        output = head(node_features, batch)
        out = head.loss(output, batch)
        assert torch.isfinite(out.loss).all()
        out.loss.sum().backward()
        optim.step()

    # Check buffers have all changed (if online_normalisation)
    for name, buffer in head.named_buffers():
        if online_normalisation:
            print(name, buffer, orig_head_buffers[name])
            assert not torch.allclose(buffer, orig_head_buffers[name])
        else:
            print(name, buffer, orig_head_buffers[name])
            assert torch.allclose(buffer, orig_head_buffers[name])

    # Check that ScalarNormalizer has learned to scale the std of our predictions.
    # We expect ~10x std, but we check for >3x to be lenient.
    # Note that, if online_normalisation=False, we don't expect our
    # predictions to have changed significantly after 5 steps and LR 1e-4.
    head.eval()
    node_features = batch.node_features[_KEY]
    pred_after = head.predict(node_features, batch)
    if online_normalisation:
        assert pred_after.std() > 3 * pred_before.std()
    else:
        assert pred_after.std() < 3 * pred_before.std()


def test_full_3x3_to_voigt_6_stress_with_ase():
    stress_matrix = torch.randn((3, 3))
    voigt = torch_full_3x3_to_voigt_6_stress(stress_matrix).numpy()
    ase_voigt = full_3x3_to_voigt_6_stress(stress_matrix.numpy())
    assert np.allclose(voigt, ase_voigt)
