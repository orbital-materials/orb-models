import pytest

from orb_models.forcefield import pretrained


@pytest.fixture(scope="module")
def orb_v2_and_config():
    """Load the orb-v2 model and system configuration."""
    model, sys_config = pretrained.orb_v2(device="cpu")
    return model, sys_config


@pytest.fixture(scope="module")
def orb_v3_direct_omat_and_config():
    """Load the orb model and system configuration."""
    model, sys_cfg = pretrained.orb_v3_direct_inf_omat()
    return model, sys_cfg


@pytest.fixture(scope="module")
def orb_v3_conservative_omat_and_config():
    """Load the orb model and system configuration."""
    model, sys_cfg = pretrained.orb_v3_conservative_inf_omat()
    model.loss_weights["rotational_grad"] = 1.0
    model.loss_weights["grad_stress"] = 1.0
    return model, sys_cfg


@pytest.fixture(scope="module")
def orb_v3_conservative_omol_and_config():
    """Load the orb model and system configuration."""
    model, sys_cfg = pretrained.orb_v3_conservative_omol()
    model.loss_weights["rotational_grad"] = 1.0
    model.loss_weights["grad_stress"] = 1.0
    return model, sys_cfg


@pytest.fixture(scope="module")
def orbmol_v2_and_config():
    """Load the orbmol-v2 model with learnable electrostatics."""
    model, sys_cfg = pretrained.orbmol_v2()
    model.loss_weights["rotational_grad"] = 1.0
    model.loss_weights["grad_stress"] = 1.0
    return model, sys_cfg
