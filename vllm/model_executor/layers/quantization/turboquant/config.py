# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant configuration."""

import math
from dataclasses import dataclass

# Named TQ presets: each maps to frozen config parameters.
# key_quant_bits: 8 = FP8 keys, 3-4 = MSE (Lloyd-Max) quantized keys.
# value_quant_bits: 3-4 = uniform quantized values.
TQ_PRESETS: dict[str, dict] = {
    "turboquant_k8v4": {
        "key_quant_bits": 8,
        "value_quant_bits": 4,
        "norm_correction": False,
    },
    "turboquant_4bit_nc": {
        "key_quant_bits": 4,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_k3v4_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_3bit_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
    },
}


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV-cache quantization.

    Uses PolarQuant (WHT rotation + Lloyd-Max scalar quantization) for keys
    and uniform quantization for values. QJL is intentionally omitted —
    community consensus (5+ independent groups) found it hurts attention
    quality by amplifying variance through softmax.

    Named presets (use via --kv-cache-dtype):
        turboquant_k8v4:   FP8 keys + 4-bit values, 2.6x, +1.17% PPL
        turboquant_4bit_nc: 4-bit MSE keys + 4-bit values + NC, 3.8x, +2.71%
        turboquant_k3v4_nc: 3-bit MSE keys + 4-bit values + NC, ~3.5x, +10.63%
        turboquant_3bit_nc: 3-bit MSE keys + 3-bit values + NC, 4.9x, +20.59%

    Args:
        head_dim: Attention head dimension (e.g. 64, 96, 128).
        key_quant_bits: Bits for key quantization. 8 = FP8 keys (no
            rotation/MSE). 3-4 = Lloyd-Max MSE quantized keys.
        value_quant_bits: Bits per value dimension for uniform quantization.
            3 = 8 levels, 4 = 16 levels (default).
        seed: Base seed for deterministic random matrix generation.
            Actual seed per layer = seed + layer_idx * 1337.
        norm_correction: Re-normalize centroid vectors to unit norm before
            inverse rotation during dequant. Fixes quantization-induced norm
            distortion, improving PPL by ~0.8% at 4-bit.
    """

    head_dim: int = 128
    key_quant_bits: int = 3  # 3-4 = MSE keys, 8 = FP8 keys
    value_quant_bits: int = 4  # 3-4 = uniform quantized values
    seed: int = 42
    norm_correction: bool = False

    @property
    def key_fp8(self) -> bool:
        """Whether keys are stored as FP8 — no rotation/quantization needed."""
        return self.key_quant_bits == 8

    @property
    def mse_bits(self) -> int:
        """MSE quantizer bit-width (determines centroid count: 2^mse_bits).

        For MSE key modes, equals key_quant_bits.
        For FP8 key mode, falls back to value_quant_bits (centroids are still
        needed for continuation-prefill dequant and decode kernel params).
        """
        if self.key_fp8:
            return self.value_quant_bits
        return self.key_quant_bits

    @property
    def key_mse_bits(self) -> int:
        """MSE bits actually used for key quantization (0 if FP8 keys)."""
        if self.key_fp8:
            return 0
        return self.key_quant_bits

    @property
    def centroid_bits(self) -> int:
        """Bits for centroid generation — always non-zero."""
        return self.mse_bits

    @property
    def n_centroids(self) -> int:
        return 2**self.mse_bits

    @property
    def key_packed_size(self) -> int:
        """Packed bytes for a single KEY vector.

        FP8 mode (key_quant_bits=8):
          head_dim bytes (1 byte per element, no overhead).

        TQ mode:
          - MSE indices: ceil(head_dim * key_mse_bits / 8) bytes
          - vec_norm:     2 bytes (float16)
          - res_norm:     2 bytes (float16)
        """
        if self.key_fp8:
            return self.head_dim  # 1 byte per element
        mse_bytes = math.ceil(self.head_dim * self.key_mse_bits / 8)
        norm_bytes = 4  # 2x float16
        return mse_bytes + norm_bytes

    @property
    def effective_value_quant_bits(self) -> int:
        """Actual bits used for value storage."""
        return self.value_quant_bits

    @property
    def value_packed_size(self) -> int:
        """Packed bytes for a single VALUE vector.

        Uniform quantization: ceil(head_dim * bits / 8) + 4 bytes (scale + zero fp16).
        """
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4  # +2 scale(fp16) +2 zero(fp16)

    @property
    def slot_size(self) -> int:
        """Total packed bytes per head per position (key + value combined).

        Layout: [key_packed | value_packed]
        """
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        """Slot size aligned for cross-head-size page divisibility.

        Two-step alignment for Gemma4's hybrid architecture (25x SWA at
        head_dim=256, 5x global at head_dim=512, with Hk_swa=8, Hk_global=2):

        Step 1 - basic alignment:
          Pad to next multiple of max(2, head_dim//64) for natural word
          alignment.
            head=256: align=4 -> slot unchanged if already divisible
            head=512: align=8 -> slot unchanged if already divisible

        Step 2 - cross-layer page divisibility (head_dim=512 only):
          vLLM's unify_kv_cache_spec_page_size requires page sizes across
          all layers to be mutually divisible.  Page for a layer =
          block_size x num_kv_heads x slot_aligned.  For page_swa to be
          divisible by page_global we need:
            (Hk_swa x slot_256) % (Hk_global x slot_512_aligned) == 0
          i.e. slot_512_aligned must be a divisor of
            (Hk_swa / Hk_global) x slot_256  =  4 x slot_256.
          The simplest sufficient condition that holds for all Gemma4 presets
          is: slot_512_aligned is a multiple of slot_256_aligned (n==2 for all
          current presets, giving page ratio 8xslot_256 / (2x2xslot_256) = 2).

        Preset examples (k8v4 / 4bit_nc / k3v4_nc / 3bit_nc):
          head=256 slot: 388 / 264 / 232 / 200
          head=512 raw:  772 / 520 / 456 / 392
          head=512 after step-1: 776 / 520 / 456 / 392
          head=512 final: 776 / 528 / 464 / 400   (= 2x the head=256 value)
        """
        # Validated head_dim values: {256, 512}. Others emit a warning.
        _VALIDATED = {256, 512}
        if self.head_dim not in _VALIDATED:
            import warnings
            warnings.warn(
                f"TurboQuant slot_size_aligned: head_dim={self.head_dim} is "
                f"not in validated set {_VALIDATED}. "
                "Verify page size divisibility manually before enabling TurboQuant.",
                UserWarning, stacklevel=2,
            )

        # Step 1: basic word alignment.
        s = self.slot_size
        alignment = max(2, self.head_dim // 64)
        remainder = s % alignment
        s = s if remainder == 0 else s + (alignment - remainder)

        # Step 2: for head_dim=512, ensure s is a multiple of the head_dim=256
        # slot so that page sizes satisfy the divisibility constraint across
        # Gemma4's hybrid SWA/global layer types.
        if self.head_dim == 512:
            ref_slot = TurboQuantConfig(
                head_dim=256,
                key_quant_bits=self.key_quant_bits,
                value_quant_bits=self.value_quant_bits,
                norm_correction=self.norm_correction,
            ).slot_size_aligned
            remainder2 = s % ref_slot
            if remainder2 != 0:
                s += ref_slot - remainder2

        return s

    @staticmethod
    def get_boundary_skip_layers(num_layers: int, n: int = 2) -> list[str]:
        """Get layer indices to skip TQ compression (boundary protection).

        Returns first N and last N layer indices as strings, suitable for
        kv_cache_dtype_skip_layers.
        """
        if n <= 0 or num_layers <= 0:
            return []
        n = min(n, num_layers // 2)  # don't skip more than half
        first = list(range(n))
        last = list(range(num_layers - n, num_layers))
        # Deduplicate (if num_layers <= 2*n)
        indices = sorted(set(first + last))
        return [str(i) for i in indices]

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> "TurboQuantConfig":
        """Create config from a named preset.

        Valid presets: turboquant_k8v4, turboquant_4bit_nc, etc.
        """
        if cache_dtype not in TQ_PRESETS:
            valid = ", ".join(TQ_PRESETS.keys())
            raise ValueError(
                f"Unknown TurboQuant cache dtype: {cache_dtype!r}. "
                f"Valid presets: {valid}"
            )
        preset = TQ_PRESETS[cache_dtype]
        return TurboQuantConfig(
            head_dim=head_dim,
            key_quant_bits=preset["key_quant_bits"],
            value_quant_bits=preset["value_quant_bits"],
            norm_correction=preset["norm_correction"],
        )
