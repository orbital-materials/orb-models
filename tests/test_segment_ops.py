"""Tests for the mini graph library forked from Jraph."""

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from orb_models.forcefield import segment_ops


@pytest.mark.parametrize(
    ("reduction, dtype"),
    [
        # NOTE: mean is only supported for floats, not ints
        ("mean", torch.float32),
        ("mean", torch.float64),
        ("sum", torch.float32),
        ("sum", torch.float64),
        ("sum", torch.int32),
        ("sum", torch.int64),
        ("max", torch.float32),
        ("max", torch.float64),
        ("max", torch.int32),
        ("max", torch.int64),
    ],
)
def test_aggregate_nodes(reduction, dtype):
    tensor = torch.arange(120).view(12, 10)
    tensor = tensor.to(dtype=dtype)
    sizes = torch.tensor([3, 5, 4], dtype=torch.long)

    res = segment_ops.aggregate_nodes(tensor, sizes, reduction=reduction)

    assert res.dtype == dtype
    assert res.shape == (3, 10)

    if reduction == "sum":
        # summing in torch changes int to long! (https://github.com/pytorch/pytorch/issues/115832)
        reduce_fn = lambda x: x.sum(dim=0, dtype=x.dtype)
    elif reduction == "mean":
        reduce_fn = lambda x: x.mean(dim=0)
    elif reduction == "max":
        reduce_fn = lambda x: x.max(dim=0)[0]

    assert torch.allclose(res[0, :], reduce_fn(tensor[:3, :]))
    assert torch.allclose(res[1, :], reduce_fn(tensor[3:8, :]))
    assert torch.allclose(res[2, :], reduce_fn(tensor[8:, :]))


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_unsorted_segment_sum(dtype):
    data = torch.arange(5)
    data = data.to(dtype=dtype)
    ids = torch.tensor([0, 1, 1, 2, 2])

    result = segment_ops.segment_sum(data, ids, 3)
    assert result.dtype == dtype
    assert result.tolist() == [0, 3, 7]


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_unsorted_segment_sum_2d(dtype):
    # grid of range 0-20, in rows of 4
    data = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19]]
    )
    data = data.to(dtype=dtype)
    ids = torch.tensor([0, 1, 1, 2, 2])

    result = segment_ops.segment_sum(data, ids, 3)
    assert result.dtype == dtype
    assert result.tolist() == [
        [0, 1, 2, 3],  # segment 0 should be first row only
        [4 + 8, 5 + 9, 6 + 10, 7 + 11],  # segment 1 should be rows 1 and 2 summed
        [12 + 16, 13 + 17, 14 + 18, 15 + 19],  # segment 2 should be rows 3 and 4 summed
    ]


def test_segment_max():
    data = torch.tensor(
        [
            [2.3696],
            [2.3693],
            [2.3718],
            [1.6104],
            [-1.8326],
            [2.3692],
            [1.6105],
            [-1.8913],
        ]
    ).view(-1)

    ids = torch.tensor([2, 1, 0, 2, 2, 0, 1, 1])

    result = segment_ops.segment_max(data.float(), ids, 3)
    assert torch.allclose(
        result,
        torch.tensor([2.371799945831299, 2.36929988861084, 2.3696000576019287]),
    )


def test_segment_mean():
    data = torch.arange(5, dtype=torch.float)
    print(data)
    ids = torch.tensor([0, 1, 1, 2, 2])
    result = segment_ops.segment_mean(data, ids, 3)
    assert result.tolist() == [0, 1.5, 3.5]


def test_segment_softmax():
    data = torch.arange(5)
    ids = torch.tensor([0, 1, 1, 2, 2])
    result = segment_ops.segment_softmax(data, ids, 3)
    assert result.shape == data.shape

    # check all segments are normalized
    assert result[0] == 1.0
    assert result[1:3].sum() == 1.0
    assert result[3:].sum() == 1.0
