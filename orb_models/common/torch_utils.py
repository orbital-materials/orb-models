import torch


def get_device(
    requested_device: torch.device | str | int | None = None,
) -> torch.device:
    """Get a torch device, defaulting to gpu if available."""
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def to_numpy(x):
    """If x is a tensor, convert it to a float (if 1 element) or np array (if > 1 element)."""
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().numpy()
    return x


def torch_lexsort(
    keys: list[torch.Tensor],
    dim: int = -1,
    descending: bool = False,
) -> torch.Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    This is a torch version of numpy.lexsort, cribbed from
    https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/utils/lexsort.html

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1
    kwargs = dict(dim=dim, descending=descending, stable=True)
    out = keys[0].argsort(**kwargs)  # type: ignore
    for k in keys[1:]:
        out = out.gather(dim, k.gather(dim, out).argsort(**kwargs))  # type: ignore

    return out


def replace_tensor_elements_within_tolerance(
    x: torch.Tensor,
    from_val: int | float | torch.Tensor,
    to_val: int | float | torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> torch.Tensor:
    """Replace all elements of x that are close to from_val with to_val.

    Args:
        x: The source tensor to map
        from_val: The value to map from
        to_val: The value to map to
        rtol: The relative tolerance
        atol: The absolute tolerance

    Returns:
        The mapped value
    """
    if isinstance(from_val, (int, float)):
        from_val = torch.tensor(from_val, dtype=x.dtype, device=x.device)
    if isinstance(to_val, (int, float)):
        to_val = torch.tensor(to_val, dtype=x.dtype, device=x.device)

    is_value = torch.isclose(x, from_val, rtol=rtol, atol=atol)
    return torch.where(is_value, to_val, x)
