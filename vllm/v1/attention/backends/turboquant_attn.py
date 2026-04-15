# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for vLLM.

Prefill: Standard scaled dot-product attention on uncompressed K/V,
         then quantize K and store K+V into combined cache slot.
Decode:  Compute TQ attention scores from compressed cache,
         unpack FP16 values, softmax + weighted sum.

Cache layout (no leading 2 dimension):
  (num_blocks, block_size, num_kv_heads, slot_size)
  where slot_size = key_packed_size + value_fp16_size

Per-head per-position slot layout:
  [key_packed (kps bytes) | value_fp16 (D*2 bytes)]
  For turboquant_k3v4_nc head_dim=256: [100 bytes key | 512 bytes value] = 612
"""

import functools
import math
import os
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.triton_utils import triton
from vllm.utils.torch_utils import aux_stream
from vllm.v1.attention.ops.triton_turboquant_decode import (
    _tq_full_dequant_kv,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store

# CUDA stream overlap: disabled by default — degrades TTFT under concurrent
# load (489ms vs 338ms). Enable via TQ_STREAM_OVERLAP=1 for experimentation.
_USE_STREAM_OVERLAP = os.environ.get("TQ_STREAM_OVERLAP", "0") == "1"

# Continuation prefill: for small continuation chunks (q_len ≤ threshold),
# use the TQ decode kernel directly instead of full-dequant + flash_attn.
# do_kv_cache_update already stored all tokens to TQ cache, so the decode
# kernel can read them efficiently. This avoids O(cached_len) dequant work
# per continuation, eliminating the O(N²/chunk_size) collapse at long context.
_CONTINUATION_DECODE_THRESHOLD = 128

# Maximum head_dim supported by flash_attn on this deployment (SM89/FA2).
# FA2 caps at 256; FA3 on H100 also caps at 256 for standard distributions.
# Update this constant if a future flash_attn release supports larger head dims,
# so all dependent code paths are updated in one place.
_FA_MAX_HEAD_DIM = 256

from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.fa_utils import (
    is_flash_attn_varlen_func_available,
)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills


@functools.cache
def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), built on CPU.

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend using TurboQuant KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = False

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "turboquant_k8v4",
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    ]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TurboQuantMetadataBuilder"]:
        return TurboQuantMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "turboquant_4bit_nc",
    ) -> tuple[int, ...]:
        """Combined K+V cache shape — no leading 2 dimension.

        Standard attention backends use (2, num_blocks, block_size, num_kv_heads,
        head_dim) with a leading 2 to separate K and V. TurboQuant packs K+V
        into a single interleaved slot per head per position, so the cache is:

            (num_blocks, block_size, num_kv_heads, slot_size_aligned)

        Each slot = [key_packed | value_packed | padding].
        This is safe because TQ has its own get_kv_cache_shape override and
        never shares cache tensors with other backends. Layers that fall back
        to native dtype via kv_cache_dtype_skip_layers get their own
        standard-shaped cache allocation.

        head_size is the model's real head_dim. slot_size_aligned is computed
        from the TQ config to ensure correct cache allocation for all head dims.
        """
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("turboquant_")

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # head_size from spec is effective_head_size (padded_slot//2),
        # not the model's actual head_dim. Accept any positive value.
        return head_size > 0


@dataclass
class TurboQuantMetadata(AttentionMetadata):
    """Metadata for TurboQuant attention."""

    seq_lens: torch.Tensor  # (num_reqs,) — total context length per request
    slot_mapping: torch.Tensor  # (num_tokens,) — cache slot for each token
    block_table: torch.Tensor  # (num_reqs, max_num_blocks)
    query_start_loc: torch.Tensor  # (num_reqs + 1,) — cu_seqlens for queries
    num_actual_tokens: int = 0  # actual tokens (excluding padding)
    max_query_len: int = 0  # longest query in batch
    max_seq_len: int = 0  # longest context in batch
    is_prefill: bool = False
    num_decodes: int = 0  # number of decode requests (first in batch)
    num_decode_tokens: int = 0  # tokens from decode requests


class TurboQuantMetadataBuilder(AttentionMetadataBuilder[TurboQuantMetadata]):
    """Builds TurboQuantMetadata from scheduler output."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TurboQuantMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # Set seq_lens to 1 so CUDA graph capture is fast
        # (real seq_lens are filled at replay time).
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        """Build TurboQuantMetadata from common attention metadata."""
        cam = common_attn_metadata

        # With reorder_batch_threshold=1, the model runner guarantees
        # decodes come first in the batch. split_decodes_and_prefills
        # finds the boundary (operates on CPU tensors — no GPU sync).
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )

        return TurboQuantMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )


class TurboQuantAttentionImpl(AttentionImpl["TurboQuantMetadata"]):
    """TurboQuant attention implementation.

    Vectorized PyTorch: batch quantize/store, vectorized bit-unpack
    decode with einsum scores and value gather.
    """

    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        self.tq_config = TurboQuantConfig.from_cache_dtype(kv_cache_dtype, head_size)

        # Pre-compute kernel constants from config (avoid repeated arithmetic)
        cfg = self.tq_config
        self._mse_bytes = (
            math.ceil(head_size * cfg.key_mse_bits / 8)
            if not cfg.key_fp8
            else head_size
        )
        self._val_data_bytes = math.ceil(head_size * cfg.effective_value_quant_bits / 8)
        self._n_centroids = cfg.n_centroids if not cfg.key_fp8 else 1

        # Fixed NUM_KV_SPLITS (grid dims must be constant for cudagraph,
        # and benchmarks show no regression vs dynamic in eager mode).
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )

    def _ensure_on_device(self, layer, device):
        """One-time migration of TQ buffers to the correct device."""
        if layer._tq_signs.device != device:
            layer._tq_signs = layer._tq_signs.to(device)
            layer._tq_centroids = layer._tq_centroids.to(device)
        if not hasattr(layer, "_tq_cached"):
            D = layer._tq_signs.shape[0]
            signs = layer._tq_signs.float()

            # WHT rotation: orthonormal + self-inverse, enabling future
            # in-kernel butterfly fusion and trivial inverse for continuation.
            H = _build_hadamard(D, str(device))
            layer._tq_PiT = (signs.unsqueeze(1) * H).contiguous()
            layer._tq_Pi = layer._tq_PiT.T.contiguous()

            c = layer._tq_centroids.float()
            # Precompute midpoints for threshold-based quantization
            c_sorted, _ = c.sort()
            layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
            # Decode buffers are lazily allocated on first decode call.
            # With fixed NUM_KV_SPLITS (cudagraph mode), the first warmup
            # allocates them and subsequent captures reuse via buf_holder.
            layer._tq_mid_o_buf = None
            layer._tq_output_buf = None
            layer._tq_lse_buf = None
            layer._tq_cached = True

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store compressed K/V into the combined TQ cache.

        Called as a separate custom op (unified_kv_cache_update) BEFORE
        the attention forward, matching FlashAttention's split pattern.
        slot_mapping is already sliced to num_actual_tokens by the caller.

        With stream overlap enabled, the store runs on a secondary CUDA
        stream so it can overlap with the next layer's forward pass.
        """
        N = slot_mapping.shape[0]
        if N <= 0:
            return

        device = key.device
        self._ensure_on_device(layer, device)

        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        # Use stream overlap only when not capturing CUDA graphs
        stream = aux_stream() if _USE_STREAM_OVERLAP else None
        use_overlap = (
            stream is not None and not torch.cuda.is_current_stream_capturing()
        )

        if use_overlap:
            # Wait for any previous store to finish before starting new one
            torch.cuda.current_stream(device).wait_stream(stream)

            # Launch store on secondary stream
            with torch.cuda.stream(stream):
                self._store_kv(k, v, kv_cache, slot_mapping, layer._tq_centroids, layer)
        else:
            self._store_kv(k, v, kv_cache, slot_mapping, layer._tq_centroids, layer)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "TurboQuantMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]

        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Slice to actual tokens
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        q = query[:N].view(N, self.num_heads, self.head_size)

        # Get TQ buffers, ensure on device (one-time migration)
        device = q.device
        self._ensure_on_device(layer, device)
        Pi = layer._tq_Pi
        PiT = layer._tq_PiT
        centroids = layer._tq_centroids

        # Ensure any async store has completed before decode reads cache
        if (
            _USE_STREAM_OVERLAP
            and not attn_metadata.is_prefill
            and not torch.cuda.is_current_stream_capturing()
        ):
            stream = aux_stream()
            if stream is not None:
                torch.cuda.current_stream(device).wait_stream(stream)

        # Compute attention (KV cache was already updated by do_kv_cache_update)
        # With reorder_batch_threshold=1, decodes come first in the batch.
        # num_decodes/num_decode_tokens from metadata give the split point.
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            # Pure decode batch — fast path
            attn_out = self._decode_attention(
                q, kv_cache, attn_metadata, Pi, centroids, PiT, layer
            )
        elif num_decodes == 0:
            # Pure prefill batch
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out = self._prefill_attention(
                q, k, v, kv_cache, attn_metadata, Pi, centroids, PiT
            )
        else:
            # Mixed batch: decodes first (guaranteed by reorder_batch).
            attn_out = torch.zeros(
                N, self.num_heads, self.head_size, device=device, dtype=q.dtype
            )

            # --- Decode portion (first num_decodes requests) ---
            # Use full-batch max_seq_len as safe upper bound (no GPU sync).
            decode_meta = TurboQuantMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
            )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, Pi, centroids, PiT, layer
            )

            # --- Prefill portion (remaining requests) ---
            # CRITICAL: use prefill-specific max_seq_len so flash_attn's
            # fast path (max_query_len == max_seq_len) triggers for
            # first-chunk prefills. Using full-batch max_seq_len breaks
            # this because decode requests inflate max_seq_len.
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            prefill_max_seq = prefill_seq_lens.max().item()
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_meta = TurboQuantMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                Pi,
                centroids,
                PiT,
            )

        # Write into output buffer: attn_out is (N, Hq, D)
        # output may be 2D (N, Hq*D) or 3D (N, Hq, D)
        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    # ------------------------------------------------------------------ #
    #  Store K/V into combined cache (vectorized)                         #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        slot_mapping: torch.Tensor,
        centroids: torch.Tensor,
        layer: "AttentionLayer",
    ):
        """Quantize + store via fused Triton kernel."""
        triton_turboquant_store(
            key,
            value,
            kv_cache,
            slot_mapping,
            layer._tq_PiT,
            centroids,
            layer._tq_midpoints,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
        )

    # ------------------------------------------------------------------ #
    #  Prefill: SDPA on raw Q/K/V with causal mask                        #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, Hq, D = query.shape

        # Fast path: use flash_attn for first-chunk prefills (all K/V in batch).
        # max_query_len == max_seq_len means no request has prior cached KV.
        # Both are Python ints — no GPU sync.
        if _HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len and D <= _FA_MAX_HEAD_DIM:
            output = torch.empty(N, Hq, D, device=query.device, dtype=query.dtype)
            flash_attn_varlen_func(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
                softmax_scale=self.scale,
                causal=True,
                out=output,
            )
            return output

        # Continuation or no flash_attn: per-request attention.
        # For continuation chunks (seq_len > q_len), we must attend to
        # previously cached K/V from the TQ cache, not just the current
        # chunk's raw K/V.
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)

        # Convert to Python lists once (single CPU-GPU sync) instead of
        # per-request .item() calls that each force a sync.
        qsl = query_start_loc.tolist()
        seq_lens_list = attn_metadata.seq_lens.tolist()

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V are in the current batch.
                if _HAS_FLASH_ATTN and D <= _FA_MAX_HEAD_DIM:
                    out = torch.empty_like(q_seq)
                    cu = torch.tensor(
                        [0, q_len], device=query.device, dtype=torch.int32
                    )
                    flash_attn_varlen_func(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=cu,
                        cu_seqlens_k=cu,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                        softmax_scale=self.scale,
                        causal=True,
                        out=out,
                    )
                else:
                    q_t = q_seq.transpose(0, 1).contiguous()
                    k_t = k_seq.transpose(0, 1).contiguous()
                    v_t = v_seq.transpose(0, 1).contiguous()
                    out = F.scaled_dot_product_attention(
                        q_t,
                        k_t,
                        v_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
            else:
                # Continuation chunk: tokens already stored to TQ cache
                # by do_kv_cache_update. Use decode kernel directly to
                # avoid O(cached_len) full-dequant per continuation.
                # For large continuations, fall back to _continuation_prefill.
                cached_len = seq_len - q_len
                if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                    # Fast path: treat each query as a decode request
                    # with incremental seq_lens for causal masking.
                    synth_seq_lens = torch.arange(
                        cached_len + 1,
                        seq_len + 1,
                        device=query.device,
                        dtype=attn_metadata.seq_lens.dtype,
                    )
                    synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                    out = triton_turboquant_decode_attention(
                        query=q_seq,
                        kv_cache=kv_cache,
                        block_table=synth_bt,
                        seq_lens=synth_seq_lens,
                        Pi=Pi,
                        centroids=centroids,
                        scale=self.scale,
                        mse_bits=self.tq_config.key_mse_bits,
                        key_packed_size=self.tq_config.key_packed_size,
                        value_quant_bits=(self.tq_config.effective_value_quant_bits),
                        key_fp8=self.tq_config.key_fp8,
                        norm_correction=self.tq_config.norm_correction,
                        PiT=PiT,
                    )
                else:
                    # Large continuation: dequant cached K/V and use
                    # flash_attn for better throughput.
                    out = self._continuation_prefill(
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        attn_metadata.block_table[i : i + 1],
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                    )
                output[q_start:q_end] = out.to(query.dtype)

        return output

    def _continuation_prefill(
        self,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
    ) -> torch.Tensor:
        """Handle continuation chunk by dequanting cached K/V from TQ cache.

        Dequants previously cached K/V, concatenates with the current
        chunk's raw K/V, then runs flash_attn with causal masking.
        """
        # This function uses Python for-loops and dynamic torch.arange calls
        # whose shapes depend on runtime seq_len. It is incompatible with CUDA
        # graph capture and must only be called from the prefill path (which
        # vLLM never captures with AttentionCGSupport.UNIFORM_BATCH).
        assert not torch.cuda.is_current_stream_capturing(), (
            "_continuation_prefill cannot run inside CUDA graph capture"
        )

        q_len, Hq, D = query.shape
        Hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]
        BLOCK_D = triton.next_power_of_2(D)

        mse_bytes = self._mse_bytes
        val_data_bytes = self._val_data_bytes
        n_centroids = self._n_centroids

        # Dequant cached K/V from TQ cache
        # Allocate slightly over to align to block_size for the grid
        alloc_len = math.ceil(cached_len / block_size) * block_size
        k_cached = torch.zeros(1, Hk, alloc_len, D, dtype=torch.float16, device=device)
        v_cached = torch.zeros(1, Hk, alloc_len, D, dtype=torch.float16, device=device)

        grid = (alloc_len, 1 * Hk)
        _tq_full_dequant_kv[grid](
            kv_cache,
            block_table,
            centroids.float(),
            k_cached,
            v_cached,
            k_cached.stride(0),
            k_cached.stride(1),
            k_cached.stride(2),
            v_cached.stride(0),
            v_cached.stride(1),
            v_cached.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_HEADS=Hk,
            MSE_BYTES=mse_bytes,
            KPS=self.tq_config.key_packed_size,
            VQB=self.tq_config.effective_value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            MSE_BITS=self.tq_config.key_mse_bits,
            N_CENTROIDS=n_centroids,
            KEY_FP8=1 if self.tq_config.key_fp8 else 0,
            BLOCK_D=BLOCK_D,
            NORM_CORRECTION=1 if self.tq_config.norm_correction else 0,
            FP8_E4B15=_use_fp8_e4b15(device.index or 0),
            num_warps=4,
        )

        # Inverse-rotate MSE keys back to original space
        if not self.tq_config.key_fp8:
            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D).float()
            k_flat = k_flat @ Pi.float()
            k_cached_trim = (
                k_flat.to(torch.float16).reshape(Hk, cached_len, D).transpose(0, 1)
            )  # (cached_len, Hk, D)
        else:
            k_cached_trim = (
                k_cached[0, :, :cached_len, :].transpose(0, 1).contiguous()
            )  # (cached_len, Hk, D)
        del k_cached

        v_cached_trim = (
            v_cached[0, :, :cached_len, :].transpose(0, 1).contiguous()
        )  # (cached_len, Hk, D)
        del v_cached

        # Attention: q_len queries attending to seq_len K/V with causal mask.
        # NOTE: For D > _FA_MAX_HEAD_DIM we deliberately avoid the upfront
        # k_full/v_full concat to prevent OOM at large contexts.
        # k_cached_trim is float16 (Triton output); qdtype is typically bf16.
        # A .to(qdtype) on (96K, 2, 512) creates a 188 MB copy.  Allocating
        # k_full THEN v_full while k_full is live requires ~376 MB new, but
        # only ~260 MB is free on RTX 4090 with 96K context.
        # For D <= _FA_MAX_HEAD_DIM (SWA layers, Hk=8) the footprint is half
        # and flash_attn requires contiguous tensors, so we keep the concat.
        qdtype = query.dtype

        if _HAS_FLASH_ATTN and D <= _FA_MAX_HEAD_DIM:
            k_full = torch.cat([k_cached_trim.to(qdtype), key_chunk], dim=0)
            del k_cached_trim
            v_full = torch.cat([v_cached_trim.to(qdtype), val_chunk], dim=0)
            del v_cached_trim
            output = torch.empty(q_len, Hq, D, device=device, dtype=qdtype)
            cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
            cu_seqlens_k = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
            flash_attn_varlen_func(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
                out=output,
            )
            return output
        else:
            # D > _FA_MAX_HEAD_DIM: FA2 unavailable on SM89 (max head_dim=256).
            # FA3 on H100 also caps at 256 for standard distributions; update
            # _FA_MAX_HEAD_DIM if a future flash_attn release supports larger dims.
            #
            # Manual FlashAttention-style online softmax with chunked Q and K.
            # Only CHUNK_K rows of K are materialized at a time, so peak extra
            # allocation stays << 16 MB regardless of context length.
            #
            # K/V segmentation (avoids upfront concat):
            #   positions [0, cached_len)       -> k_cached_trim / v_cached_trim
            #   positions [cached_len, seq_len) -> key_chunk / val_chunk
            # Dtype conversion happens per chunk (lazy), not upfront.
            #
            # Algorithm (per Q chunk [qi:qj]):
            #   m  = -inf    (running max over K, per query)
            #   l  = 0       (running softmax denominator)
            #   acc= 0       (running weighted V sum)
            #   for K chunk [ki:kj]:
            #     scores = Q_c @ K_c^T * scale        (float32)
            #     causal mask: score[r,c] = -inf if ki+c > qi+r+cached_len
            #     chunk_max  = max(scores, dim=-1)
            #     m_new      = max(m, chunk_max)
            #     l   = exp(m-m_new)*l + sum(exp(scores-m_new), dim=-1)
            #     acc = exp(m-m_new)*acc + exp(scores-m_new) @ V_c
            #     m   = m_new
            #   out[qi:qj] = acc / l
            head_ratio = Hq // Hk if Hk < Hq else 1

            q_t = query.permute(1, 0, 2).contiguous()   # (Hq, q_len, D)

            # Tile sizes: keep expanded K chunk <= 16 MB, scores <= 8 MB.
            # K expand: head_ratio x CHUNK_K x D x 2 <= 16 MB
            #   -> CHUNK_K <= 16 MB / (head_ratio x D x 2)
            # Scores:   Hq x CHUNK_Q x CHUNK_K x 4   <= 8 MB
            #   -> CHUNK_Q x CHUNK_K <= 8 MB / (Hq x 4)
            # The min(512, ...) cap bounds the K-chunk grid size and register
            # pressure in the GQA expand step; 512 is the effective ceiling
            # for D >= 128 where the memory formula alone gives > 512.
            CHUNK_K = max(1, min(512, (16 * 1024 * 1024) // (head_ratio * D * 2)))
            CHUNK_Q = max(1, min(q_len, (8 * 1024 * 1024) // (Hq * CHUNK_K * 4)))

            out = torch.empty_like(q_t)  # (Hq, q_len, D)

            for qi in range(0, q_len, CHUNK_Q):
                qj = min(qi + CHUNK_Q, q_len)
                cs_q = qj - qi
                q_c = q_t[:, qi:qj, :]  # (Hq, cs_q, D)

                # Online softmax state (float32 for numerical stability)
                m   = q_c.new_full((Hq, cs_q, 1), float('-inf'), dtype=torch.float32)
                l   = q_c.new_zeros((Hq, cs_q, 1), dtype=torch.float32)
                acc = q_c.new_zeros((Hq, cs_q, D), dtype=torch.float32)

                # K-loop upper bound: the last query in this chunk (at index qj-1)
                # can attend to K positions 0..qj-1+cached_len = max_k_pos-1.
                # Using qj (not qi) is intentional -- earlier queries in the chunk
                # are correctly restricted by the per-position causal mask below,
                # not by this loop bound.
                max_k_pos = qj + cached_len

                for ki in range(0, max_k_pos, CHUNK_K):
                    kj = min(ki + CHUNK_K, max_k_pos)
                    cs_k = kj - ki

                    # Source K/V from the correct segment (no upfront concat):
                    #   [0, cached_len)       -> TQ-decompressed cache (float16)
                    #   [cached_len, seq_len) -> current input chunk (qdtype)
                    # The boundary case (straddles both) fires at most once per
                    # Q-chunk and allocates only CHUNK_K rows.
                    if kj <= cached_len:
                        k_c_seq = k_cached_trim[ki:kj].to(qdtype)  # (cs_k, Hk, D)
                        v_c_seq = v_cached_trim[ki:kj].to(qdtype)
                    elif ki >= cached_len:
                        off = ki - cached_len
                        k_c_seq = key_chunk[off:off + cs_k]         # already qdtype
                        v_c_seq = val_chunk[off:off + cs_k]
                    else:
                        # Boundary chunk: straddles cached and current tokens.
                        k_c_seq = torch.cat([
                            k_cached_trim[ki:cached_len].to(qdtype),
                            key_chunk[:kj - cached_len],
                        ], dim=0)
                        v_c_seq = torch.cat([
                            v_cached_trim[ki:cached_len].to(qdtype),
                            val_chunk[:kj - cached_len],
                        ], dim=0)

                    # Expand K/V from Hk to Hq heads (GQA)
                    # k_c_hk: (Hk, cs_k, D) -> (Hq, cs_k, D)
                    k_c_hk = k_c_seq.permute(1, 0, 2)
                    v_c_hk = v_c_seq.permute(1, 0, 2)
                    if head_ratio > 1:
                        k_c = k_c_hk.repeat_interleave(head_ratio, dim=0)
                        v_c = v_c_hk.repeat_interleave(head_ratio, dim=0)
                    else:
                        k_c = k_c_hk
                        v_c = v_c_hk                                 # (Hq, cs_k, D)

                    # Scores: (Hq, cs_q, cs_k) in float32
                    scores = torch.matmul(
                        q_c.float(), k_c.float().transpose(-1, -2)
                    ) * self.scale  # (Hq, cs_q, cs_k)

                    # Causal mask: query qi+r can attend to k ki+c iff ki+c <= qi+r+cached_len
                    # Equivalently: c <= r + (qi - ki) + cached_len
                    r = torch.arange(cs_q, device=device, dtype=torch.int32)
                    c = torch.arange(cs_k, device=device, dtype=torch.int32)
                    causal_limit = (qi - ki) + cached_len
                    # mask[r, c] = True means MASKED (future token)
                    mask = c.unsqueeze(0) > r.unsqueeze(1) + causal_limit
                    scores.masked_fill_(mask.unsqueeze(0), float('-inf'))

                    # Online softmax update.
                    # When a K chunk is entirely masked (chunk_max = -inf), we
                    # replace it with the running m so correction = exp(0) = 1
                    # and exp_scores = 0, leaving the running state unchanged.
                    # NaN safety: this substitution is safe because max_k_pos
                    # guarantees the first K chunk (ki=0) always has valid
                    # positions for every query (K position 0 <= cached_len for
                    # any query), so m is finite before any all-masked chunk
                    # can appear.
                    chunk_max = scores.max(dim=-1, keepdim=True).values  # (Hq, cs_q, 1)
                    chunk_max = torch.where(
                        torch.isinf(chunk_max), m, chunk_max
                    )  # skip all-masked chunks
                    exp_scores = torch.exp(scores - chunk_max)  # (Hq, cs_q, cs_k)

                    m_new = torch.maximum(m, chunk_max)
                    correction = torch.exp(m - m_new)  # (Hq, cs_q, 1)
                    l   = correction * l + exp_scores.sum(dim=-1, keepdim=True)
                    acc = correction * acc + torch.matmul(
                        exp_scores, v_c.float()
                    )  # (Hq, cs_q, D)
                    m = m_new

                # Normalize and write output
                out[:, qi:qj, :] = (acc / l.clamp(min=1e-10)).to(qdtype)

            del k_cached_trim, v_cached_trim
            # Permute back to (q_len, Hq, D)
            return out.permute(1, 0, 2).contiguous()  # (q_len, Hq, D)
    # ------------------------------------------------------------------ #
    #  Decode: Triton TQ decode attention                                 #
    # ------------------------------------------------------------------ #
    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot_size)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # Grab cached decode buffers from the layer (lazily allocated).
        mid_o_buf = output_buf = lse_buf = None
        if layer is not None:
            mid_o_buf = getattr(layer, "_tq_mid_o_buf", None)
            output_buf = getattr(layer, "_tq_output_buf", None)
            lse_buf = getattr(layer, "_tq_lse_buf", None)

        result = triton_turboquant_decode_attention(
            query=query,
            kv_cache=kv_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            Pi=Pi,
            centroids=centroids,
            scale=self.scale,
            mse_bits=self.tq_config.key_mse_bits,
            key_packed_size=self.tq_config.key_packed_size,
            value_quant_bits=self.tq_config.effective_value_quant_bits,
            key_fp8=self.tq_config.key_fp8,
            norm_correction=self.tq_config.norm_correction,
            PiT=PiT,
            mid_o_buf=mid_o_buf,
            output_buf=output_buf,
            lse_buf=lse_buf,
            buf_holder=layer,
            max_num_kv_splits=self.max_num_kv_splits,
        )
        return result
