#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""Data generator for ST tests.

Provides pre-generated tensors and configuration templates.
Reference: tests/ut/sample/test_rejection_sampler.py PLACEHOLDER_TOKEN_ID pattern
"""

import torch

SMALL_BATCH_SHAPE = (1, 16)
MEDIUM_BATCH_SHAPE = (16, 64)
LARGE_BATCH_SHAPE = (32, 128)
DEFAULT_SEQ_LEN = 1024
DEFAULT_HIDDEN_DIM = 4096

PLACEHOLDER_TOKEN_ID = -1


class DataGenerator:
    """Data generator for ST test data."""

    _tensor_cache: dict = {}

    @classmethod
    def get_standard_tensor(cls,
                           shape_name: str,
                           dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Get pre-generated tensor by shape name.

        Args:
            shape_name: "small_batch", "medium_batch", "large_batch"
            dtype: Tensor dtype

        Returns:
            Pre-generated tensor
        """
        shapes = {
            "small_batch": SMALL_BATCH_SHAPE,
            "medium_batch": MEDIUM_BATCH_SHAPE,
            "large_batch": LARGE_BATCH_SHAPE,
        }

        shape = shapes.get(shape_name, SMALL_BATCH_SHAPE)
        cache_key = f"{shape_name}_{dtype}"

        if cache_key not in cls._tensor_cache:
            cls._tensor_cache[cache_key] = torch.randn(*shape, dtype=dtype)

        return cls._tensor_cache[cache_key].clone()

    @classmethod
    def generate_attention_input(cls,
                                 batch_size: int = 1,
                                 seq_len: int = DEFAULT_SEQ_LEN,
                                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                                 dtype: torch.dtype = torch.float16) -> dict:
        """Generate attention input tensors.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            dtype: Tensor dtype

        Returns:
            dict with input tensors
        """
        return {
            "hidden_states": torch.randn(batch_size, seq_len, hidden_dim, dtype=dtype),
            "attention_mask": torch.ones(batch_size, 1, seq_len, seq_len, dtype=dtype),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
        }

    @classmethod
    def generate_scheduler_input(cls, num_seqs: int = 4) -> dict:
        """Generate scheduler input data.

        Args:
            num_seqs: Number of sequences

        Returns:
            dict with scheduler input
        """
        return {
            "num_seqs": num_seqs,
            "seq_group_ids": list(range(num_seqs)),
        }

    @classmethod
    def generate_worker_input(cls, batch_size: int = 1) -> dict:
        """Generate worker input data.

        Args:
            batch_size: Batch size

        Returns:
            dict with worker input
        """
        return {
            "batch_size": batch_size,
            "token_ids": torch.randint(0, 1000, (batch_size, 16)),
        }

    @classmethod
    def clear_cache(cls):
        """Clear tensor cache."""
        cls._tensor_cache.clear()


def get_test_batch_sizes() -> list:
    """Get standard test batch sizes.

    Returns:
        List of batch sizes for parameterization
    """
    return [1, 16, 32]


def get_test_dtypes() -> list:
    """Get standard test dtypes.

    Returns:
        List of dtypes for parameterization
    """
    return ["float16", "bfloat16"]


def get_test_seq_lengths() -> list:
    """Get standard test sequence lengths.

    Returns:
        List of sequence lengths for parameterization
    """
    return [128, 512, 1024]


DEFAULT_BATCH_SIZES = get_test_batch_sizes()
DEFAULT_DTYPES = get_test_dtypes()
DEFAULT_SEQ_LENGTHS = get_test_seq_lengths()