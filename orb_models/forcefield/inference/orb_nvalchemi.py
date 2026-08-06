"""Orb model wrapper for the nvalchemi-toolkit framework.

Wraps Orb forcefield models as a :class:`~nvalchemi.models.base.BaseModelMixin`-compatible
wrapper for use in any :class:`~nvalchemi.dynamics.base.BaseDynamics` engine, standalone
inference, or pipeline composition.

Usage
-----
Load a pretrained Orb model::

    from orb_models.forcefield.inference.orb_nvalchemi import OrbWrapper
    import torch

    model = OrbWrapper.from_pretrained("orb-v3-conservative-omol", device=torch.device("cuda"))

Or wrap an already-loaded model::

    from orb_models.forcefield import pretrained
    orb_model, adapter = pretrained.orb_v3_conservative_omol(device="cuda")
    model = OrbWrapper(orb_model, adapter)

Pipeline composition::

    # Orb + DFT-D3 with shared autograd (conservative models)
    pipe = PipelineModelWrapper(groups=[
        PipelineGroup(steps=[orb_wrapper, dftd3_wrapper], use_autograd=True),
    ])

Notes
-----
* Conservative models declare ``autograd_outputs`` so they can participate in
  nvalchemi pipeline autograd groups. Direct models do not.
* Stress is converted from Orb's Voigt-6 notation to the full 3x3 tensor.

When to compile
---------------
``from_pretrained(compile=...)`` compiles the *model*, which is orthogonal to compiling a *pipeline*. Choose by run mode:

* **Pipeline / dynamics (production)** — keep the wrapper **uncompiled** (``compile=False``,
  the default) and let nvalchemi compile the whole step (``FusedStage.compile()``): neighbor-list
  rebuild, forward, force/stress autograd, and integrator update all fuse into one graph.
  See https://nvidia.github.io/nvalchemi-toolkit/modules/dynamics/fused_stage.html#torch-compile-support
* **Standalone inference** — ``from_pretrained(compile=True)``: forces come from the model's own autograd
  inside the compiled forward.
* **Ad-hoc compiled pipeline** — ``torch.compile(pipe)`` over an *uncompiled* wrapper, after
  ``pipe.eval()``. Fuses just the forward + force/stress autograd (no dynamics loop).
* **Training** — conservative models must stay uncompiled (``inference=False``): training needs
  double-backward, which torch.compile does not support.
"""

from collections import OrderedDict
from typing import Any, cast

import torch
from torch import nn

try:
    from nvalchemi._typing import ModelOutputs
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.models.base import (
        BaseModelMixin,
        ModelConfig,
        NeighborConfig,
        NeighborListFormat,
    )
except ImportError as e:
    raise ImportError(
        "nvalchemi is required for the OrbWrapper interface. "
        "Install it with: pip install nvalchemi-toolkit"
    ) from e

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.common.models.gns import MoleculeGNS
from orb_models.forcefield import pretrained
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.models.conservative_regressor import ConservativeForcefieldRegressor
from orb_models.forcefield.models.direct_regressor import DirectForcefieldRegressor

__all__ = ["OrbWrapper"]

_OrbModel = DirectForcefieldRegressor | ConservativeForcefieldRegressor


def _voigt_6_to_full_3x3(voigt: torch.Tensor) -> torch.Tensor:
    """Convert Voigt notation ``[*, 6]`` to full symmetric tensor ``[*, 3, 3]``."""
    xx, yy, zz, yz, xz, xy = voigt.unbind(-1)
    row0 = torch.stack([xx, xy, xz], dim=-1)
    row1 = torch.stack([xy, yy, yz], dim=-1)
    row2 = torch.stack([xz, yz, zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


class OrbWrapper(nn.Module, BaseModelMixin):
    """Wrapper for Orb models implementing the nvalchemi :class:`BaseModelMixin` interface.

    Accepts ``DirectForcefieldRegressor`` or ``ConservativeForcefieldRegressor``.

    Parameters
    ----------
    model : DirectForcefieldRegressor | ConservativeForcefieldRegressor
        An Orb forcefield model.
    atoms_adapter : ForcefieldAtomsAdapter
        The adapter associated with the model, used for cutoff radius and
        max_num_neighbors metadata.
    """

    model: _OrbModel

    def __init__(
        self,
        model: _OrbModel,
        atoms_adapter: ForcefieldAtomsAdapter,
    ) -> None:
        super().__init__()
        self.model = model
        self.atoms_adapter = atoms_adapter

        max_nn = atoms_adapter.max_num_neighbors
        if max_nn is not None and max_nn < 120:
            raise ValueError(
                f"OrbWrapper requires max_num_neighbors >= 120 (got {max_nn}). "
                f"120 captures all edges under 6 A and is effectively uncapped, "
                f"which nvalchemi's neighbor list hook assumes."
            )

        outputs = set(model.properties)
        supports_pbc = "stress" in outputs

        required_inputs: set[str] = set()
        optional_inputs = {"cell", "neighbor_list_shifts", "pbc", "charge", "spin"}

        autograd_outputs = model.autograd_derivative_keys & outputs
        autograd_inputs = frozenset({"positions"}) if autograd_outputs else frozenset()

        self.model_config = ModelConfig(
            outputs=frozenset(outputs),
            active_outputs=set(outputs),
            autograd_outputs=autograd_outputs,
            autograd_inputs=autograd_inputs,
            required_inputs=frozenset(required_inputs),
            optional_inputs=frozenset(optional_inputs),
            supports_pbc=supports_pbc,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self.cutoff,
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

    # ------------------------------------------------------------------
    # BaseModelMixin properties
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        latent_dim: int = cast(MoleculeGNS, self.model.model).latent_dim
        return {
            "node_embeddings": (latent_dim,),
        }

    @property
    def cutoff(self) -> float:
        """Interaction cutoff in Angstroms, from the atoms adapter."""
        assert self.atoms_adapter.radius is not None, "radius is not set"
        return self.atoms_adapter.radius

    # ------------------------------------------------------------------
    # Input / output adaptation
    # ------------------------------------------------------------------

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> AtomGraphs:
        """Build an Orb ``AtomGraphs`` from a nvalchemi ``Batch``.

        Reuses the neighbor list already computed by nvalchemi's
        :class:`~nvalchemi.hooks.NeighborListHook` instead of running Orb's
        own graph construction.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        device = data.positions.device
        dtype = data.positions.dtype
        B = data.num_graphs

        positions = data.positions
        atomic_numbers = data.atomic_numbers.long()
        batch_idx = data.batch_idx.long()

        neighbor_list = data.neighbor_list.long()
        senders = neighbor_list[:, 0]
        receivers = neighbor_list[:, 1]
        E = senders.shape[0]

        neighbor_list_shifts_raw = getattr(data, "neighbor_list_shifts", None)
        cell_raw = getattr(data, "cell", None)

        if cell_raw is None:
            cell = (
                torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
            )
        else:
            cell = cell_raw.to(dtype=dtype, device=device)

        if neighbor_list_shifts_raw is None:
            shifts_physical = torch.zeros(E, 3, dtype=dtype, device=device)
            unit_shifts = torch.zeros(E, 3, dtype=dtype, device=device)
        else:
            unit_shifts = neighbor_list_shifts_raw.to(dtype=dtype, device=device)
            sender_batch = batch_idx[senders]
            shifts_physical = torch.einsum("eb,ebc->ec", unit_shifts, cell[sender_batch])

        edge_vectors = positions[receivers] - positions[senders] + shifts_physical

        pbc_raw = getattr(data, "pbc", None)
        if pbc_raw is not None:
            pbc = pbc_raw.to(device=device)
            if pbc.dim() == 1:
                pbc = pbc.unsqueeze(0).expand(B, -1)
        else:
            pbc = torch.zeros(B, 3, dtype=torch.bool, device=device)

        atomic_numbers_embedding = torch.nn.functional.one_hot(atomic_numbers, num_classes=118).to(
            dtype
        )

        n_node = torch.bincount(batch_idx, minlength=B)
        sender_batch = batch_idx[senders]
        n_edge = torch.bincount(sender_batch, minlength=B)

        max_num_neighbors = self.atoms_adapter.max_num_neighbors
        assert max_num_neighbors is not None, "max_num_neighbors is not set"

        system_features: dict[str, torch.Tensor] = {"cell": cell, "pbc": pbc}
        charge = getattr(data, "charge", None)
        if charge is not None:
            system_features["total_charge"] = charge.to(dtype=dtype, device=device).view(B)
        spin = getattr(data, "spin", None)
        if spin is not None:
            system_features["spin_multiplicity"] = spin.to(dtype=dtype, device=device).view(B)

        return AtomGraphs(
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            node_features={
                "positions": positions,
                "atomic_numbers": atomic_numbers,
                "atomic_numbers_embedding": atomic_numbers_embedding,
            },
            edge_features={
                "vectors": edge_vectors,
                "unit_shifts": unit_shifts,
            },
            system_features=system_features,
            node_targets={},
            edge_targets={},
            system_targets={},
            system_id=None,
            fix_atoms=None,
            tags=None,
            radius=self.cutoff,
            max_num_neighbors=torch.full_like(n_node, fill_value=max_num_neighbors),
        )

    def adapt_output(self, raw_output: dict[str, Any], data: AtomicData | Batch) -> ModelOutputs:
        """Map Orb output dict to nvalchemi :class:`ModelOutputs`.

        Does not call ``super().adapt_output()`` because the base implementation
        filters by ``active_outputs``, which excludes forces/stress in pipeline
        autograd mode — even though analytical (non-autograd) derivatives must
        still pass through.
        """

        output: ModelOutputs = OrderedDict()

        energy = raw_output.get("energy")
        if energy is not None:
            output["energy"] = energy.unsqueeze(-1) if energy.ndim == 1 else energy

        forces = raw_output.get("forces")
        if forces is not None:
            output["forces"] = forces

        stress = raw_output.get("stress")
        if stress is not None:
            output["stress"] = _voigt_6_to_full_3x3(torch.atleast_2d(stress))

        return output

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the Orb model and return nvalchemi-formatted outputs."""
        atom_graphs = self.adapt_input(data, **kwargs)
        active = self.model_config.active_outputs & self.model_config.outputs
        raw = self.model(  # type: ignore[operator]
            atom_graphs,
            compute_forces="forces" in active,
            compute_stress="stress" in active,
            fp64_energy=True,
        )
        return self.adapt_output(raw, data)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def compute_embeddings(self, data: AtomicData | Batch, **kwargs: Any) -> AtomicData | Batch:
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        atom_graphs = self.adapt_input(data, **kwargs)
        out = self.model.model(atom_graphs)
        node_feats = out["node_features"]

        atoms_group = data._atoms_group
        if atoms_group is not None:
            atoms_group["node_embeddings"] = node_feats
        else:
            data.node_embeddings = node_feats

        return data

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: torch.device | str = torch.device("cpu"),
        *,
        compile: bool | None = False,
        inference: bool = True,
    ) -> "OrbWrapper":
        """Load a pretrained Orb model and return an :class:`OrbWrapper`.

        Parameters
        ----------
        model_name : str
            One of: ``"orbmol-v2"``, ``"orbmol-v1-conservative"``,
            ``"orbmol-v1-direct"``, ``"orb-v3-conservative-20-omat"``,
            ``"orb-v3-conservative-inf-omat"``, ``"orb-v3-direct-20-omat"``,
            ``"orb-v3-direct-inf-omat"``, ``"orb-v3-conservative-20-mpa"``,
            ``"orb-v3-conservative-inf-mpa"``, ``"orb-v3-direct-20-mpa"``,
            ``"orb-v3-direct-inf-mpa"``, ``"orb-v2"``.
        device : torch.device | str
            Target device. Defaults to CPU.
        compile : bool | None
            Whether to ``torch.compile`` the model. ``False`` (default): don't; ``True``: always;
            ``None``: device-based heuristic (see :func:`should_compile`). See the "When to
            compile" matrix in the module docstring for which to pick per run mode.
        inference : bool
            ``True`` (default): eval mode, frozen params.
            ``False``: train mode, trainable params.

        """
        _LOADERS: dict[str, Any] = {
            # orbmol-v2 (learnable electrostatics)
            "orbmol-v2": pretrained.orbmol_v2,
            # orbmol-v1 models
            "orb-v3-conservative-omol": pretrained.orb_v3_conservative_omol,
            "orb-v3-direct-omol": pretrained.orb_v3_direct_omol,
            # orbmol-v1 model aliases
            "orbmol-v1-conservative": pretrained.orbmol_v1_conservative,
            "orbmol-v1-direct": pretrained.orbmol_v1_direct,
            # most performant orb-v3 omat models
            "orb-v3-conservative-inf-omat": pretrained.orb_v3_conservative_inf_omat,
            "orb-v3-direct-inf-omat": pretrained.orb_v3_direct_inf_omat,
            # less performant orb-v3 mptraj + alexandria models
            "orb-v3-conservative-inf-mpa": pretrained.orb_v3_conservative_inf_mpa,
            "orb-v3-direct-inf-mpa": pretrained.orb_v3_direct_inf_mpa,
        }

        if model_name not in _LOADERS:
            raise ValueError(f"Unknown model: {model_name!r}. Available: {sorted(_LOADERS)}")

        model, adapter = _LOADERS[model_name](device=device, compile=compile, train=not inference)
        assert isinstance(adapter, ForcefieldAtomsAdapter)
        wrapper = cls(model=model, atoms_adapter=adapter)
        wrapper.train(not inference)
        return wrapper
