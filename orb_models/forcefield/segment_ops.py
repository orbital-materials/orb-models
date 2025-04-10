from typing import Optional

import torch

TORCHINT = [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]


def aggregate_nodes(
    tensor: torch.Tensor,
    n_node: torch.Tensor,
    reduction: str = "mean",
    deterministic: bool = False,
) -> torch.Tensor:
    """Aggregates over a tensor based on graph sizes.

    Args:
        tensor: The tensor to aggregate over.
        n_node: A 1D tensor representing the *sizes* of graphs, i.e graph.n_node.
            NOTE: these are not segment indices!  This function exists
            to convert between node sizes and segment
            indices representing those nodes.
        reduction (str, optional): The aggregation operation to use.
        Can be "sum", "mean", or "max". Defaults to "mean".
        deterministic: to use deterministic scatter algorithms.

    Returns:
        A tensor of shape (num_graphs, feature_dim).
    """
    # We have removed this check because it causes odd traces
    # in the CUDA profiler. It is not necessary because the
    # scatter functions will throw an error if the segment
    # indices are out of bounds anyway.
    # assert n_node.sum() == tensor.shape[0]

    device = tensor.device
    count = len(n_node)
    if deterministic:
        import os

        """If you are using CUDA tensors, and your CUDA version is 10.2 or greater,
        you should set the environment variable CUBLAS_WORKSPACE_CONFIG according to CUDA documentation
        https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility"""
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    segments = torch.arange(count, device=device).repeat_interleave(n_node)
    if reduction == "sum":
        return scatter_sum(tensor, segments, dim=0)
    elif reduction == "mean":
        return scatter_mean(tensor, segments, dim=0)
    elif reduction == "max":
        return segment_max(tensor, segments, count)
    else:
        raise ValueError("Invalid reduction argument. Use sum, mean or max.")


def segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int):
    """Computes index based sum over segments of a tensor."""
    return scatter_sum(data, segment_ids, dim=0, dim_size=num_segments)


def segment_max(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int):
    """Computes index based max over segments of a tensor."""
    return scatter_max(data, segment_ids, dim=0, dim_size=num_segments)


def segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int):
    """Computes index based mean over segments of a tensor."""
    return scatter_mean(data, segment_ids, dim=0, dim_size=num_segments)


def segment_softmax(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    weights: Optional[torch.Tensor] = None,
):
    """Computes a softmax over segments of the tensor.

    Note that unlike other segment reductions, this returns a tensor the
    *same* shape as `data`, because this just normalizes over sub-parts
    of a tensor.

    Args:
        data: A tensor which we want to sum over in segments.
        segment_ids: The segment indices tensor.
        num_segments: The number of segments.
        weights: Optional weights tensor to multiply exponents.

    Returns:
        A tensor of same data type as data, also of the same shape.
    """
    # subtract mean for stability
    data_max = segment_max(data, segment_ids, num_segments)
    data = data - data_max[segment_ids]

    unnormalised_probs = torch.exp(data)
    if weights is not None:
        unnormalised_probs = unnormalised_probs * weights
    denominator = segment_sum(unnormalised_probs, segment_ids, num_segments)

    return safe_division(unnormalised_probs, denominator, segment_ids)


def safe_division(
    numerator: torch.Tensor, denominator: torch.Tensor, segment_ids: torch.Tensor
):
    """Divides logits by denominator, setting 0 where the denominator is zero."""
    result = torch.where(
        denominator[segment_ids] == 0, 0, numerator / denominator[segment_ids]
    )
    return result


# The Following implementation of scatter function are taken from
# https://github.com/mir-group/pytorch_runstats/blob/main/torch_runstats/scatter.py which
# copies torch_scatter but removes the dependency.


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """Broadcasts the source tensor to match the shape of the other tensor along the specified dimension.

    Args:
        src (torch.Tensor): The source tensor to be broadcasted.
        other (torch.Tensor): The target tensor to match the shape of.
        dim (int): The dimension along which to broadcast.

    Returns:
        torch.Tensor: The broadcasted source tensor.
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    """Applies a sum reduction of the src tensor along the specified dimension.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
        dim (int, optional): The dimension along which to index. Defaults to -1.
        out (Optional[torch.Tensor], optional): The output tensor. Defaults to None.
        dim_size (Optional[int], optional): Size of the output tensor. Defaults to None.
        reduce (str, optional): The reduction operation to perform. Defaults to "sum".

    Returns:
        torch.Tensor: The output tensor with values scattered and summed.
    """
    assert reduce == "sum"
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_std(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    unbiased: bool = True,
) -> torch.Tensor:
    """Computes the standard deviation of the src tensor along the specified dimension.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
        dim (int, optional): The dimension along which to index. Defaults to -1.
        out (Optional[torch.Tensor], optional): The output tensor. Defaults to None.
        dim_size (Optional[int], optional): Size of the output tensor. Defaults to None.
        unbiased (bool, optional): Whether to use the unbiased estimation. Defaults to True.

    Returns:
        torch.Tensor: The output tensor with standard deviation values.
    """
    if out is not None:
        dim_size = out.size(dim)

    if dim < 0:
        dim = src.dim() + dim

    count_dim = dim
    if index.dim() <= dim:
        count_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

    index = _broadcast(index, src, dim)
    tmp = scatter_sum(src, index, dim, dim_size=dim_size)
    count = _broadcast(count, tmp, dim).clamp(1)
    mean = tmp.div(count)

    var = src - mean.gather(dim, index)
    var = var * var
    out = scatter_sum(var, index, dim, out, dim_size)

    if unbiased:
        count = count.sub(1).clamp_(1)
    out = out.div(count + 1e-6).sqrt()

    return out


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Computes the mean of the src tensor along the specified dimension.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
        dim (int, optional): The dimension along which to index. Defaults to -1.
        out (Optional[torch.Tensor], optional): The output tensor. Defaults to None.
        dim_size (Optional[int], optional): Size of the output tensor. Defaults to None.

    Returns:
        torch.Tensor: The output tensor with mean values.
    """
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Computes the maximum of the src tensor for each group defined by index along the specified dimension.

    Args:
        src (torch.Tensor): The source tensor.
        index (torch.Tensor): The indices of elements to scatter.
        dim (int, optional): The dimension along which to index. Defaults to -1.
        out (Optional[torch.Tensor], optional): The output tensor. Defaults to None.
        dim_size (Optional[int], optional): Size of the output tensor. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the maximum values and their indices.
    """
    if out is not None:
        raise NotImplementedError("The 'out' argument is not supported for scatter_max")

    if src.dtype in TORCHINT:
        init_value = torch.iinfo(src.dtype).min
    else:
        init_value = float("-inf")  # type: ignore

    if dim < 0:
        dim = src.dim() + dim

    if dim_size is None:
        dim_size = int(index.max()) + 1

    result = torch.empty(
        dim_size,
        *src.shape[:dim],
        *src.shape[dim + 1 :],
        dtype=src.dtype,
        device=src.device,
    )
    result.fill_(init_value)
    broadcasted_index = _broadcast(index, src, dim)
    result = result.scatter_reduce(
        dim, broadcasted_index, src, reduce="amax", include_self=False
    )

    return result
