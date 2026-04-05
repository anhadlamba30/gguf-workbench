from __future__ import annotations

import mmap
import struct
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from .constants import (
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
    GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K,
    GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K,
    GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_K,
    GGML_TYPE_IQ2_XXS,
    GGML_TYPE_IQ2_XS,
    GGML_TYPE_IQ3_XS,
    GGML_TYPE_IQ1_S,
    GGML_TYPE_IQ4_XS,
    GGML_TYPE_IQ2_S,
)

from .parser import TensorInfo


def bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    arr_u32 = arr_u16.astype(np.uint32) << 16
    return arr_u32.view(np.float32)


def f32_to_bf16(arr_f32: np.ndarray) -> np.ndarray:
    arr_u32 = np.asarray(arr_f32, dtype=np.float32).view(np.uint32)
    rounding_bias = ((arr_u32 >> 16) & 1) + 0x7FFF
    return ((arr_u32 + rounding_bias) >> 16).astype(np.uint16)


def dequantize_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    block_size = 32
    n_blocks = (n_elements + block_size - 1) // block_size
    scales = np.frombuffer(data[:n_blocks], dtype=np.float32).copy()
    quants = np.frombuffer(
        data[n_blocks : n_blocks + n_blocks * block_size], dtype=np.int8
    ).copy()
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n_elements)
        result[start:end] = scales[i] * quants[start:end].astype(np.float32)
    return result


def dequantize_q4_0(data: bytes, n_elements: int) -> np.ndarray:
    block_size = 32
    n_blocks = (n_elements + block_size - 1) // block_size
    scales = (
        np.frombuffer(data[: n_blocks * 2], dtype=np.float16).astype(np.float32).copy()
    )
    quants = np.frombuffer(
        data[n_blocks * 2 : n_blocks * 2 + n_blocks * 16], dtype=np.uint8
    ).copy()
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        scale = scales[i]
        qs = quants[i * 16 : (i + 1) * 16]
        q0 = (qs >> 4) & 0x0F
        q1 = qs & 0x0F
        q0 = np.where(q0 >= 8, q0 - 16, q0).astype(np.float32)
        q1 = np.where(q1 >= 8, q1 - 16, q1).astype(np.float32)
        result[i * 32 + 0 : i * 32 + 16] = scale * q0
        result[i * 32 + 16 : i * 32 + 32] = scale * q1
    return result[:n_elements]


def dequantize_q4_k(data: bytes, n_elements: int) -> np.ndarray:
    super_block_size = 256
    block_size = 32
    n_super_blocks = (n_elements + super_block_size - 1) // super_block_size
    result = np.zeros(n_elements, dtype=np.float32)
    offset = 0
    for _ in range(n_super_blocks):
        if offset >= len(data):
            break
        d = np.frombuffer(data[offset : offset + 2], dtype=np.float16)[0]
        d_min = np.frombuffer(data[offset + 2 : offset + 4], dtype=np.float16)[0]
        scales = np.frombuffer(data[offset + 4 : offset + 20], dtype=np.float16).astype(
            np.float32
        )
        mins = np.frombuffer(data[offset + 20 : offset + 36], dtype=np.float16).astype(
            np.float32
        )
        offset += 36
        for j in range(8):
            scale = scales[j]
            min_val = mins[j]
            q = np.frombuffer(data[offset : offset + 16], dtype=np.uint8)
            q0 = (q >> 4) & 0x0F
            q1 = q & 0x0F
            v0 = np.where(q0 >= 8, q0 - 16, q0).astype(np.float32) * scale + min_val
            v1 = np.where(q1 >= 8, q1 - 16, q1).astype(np.float32) * scale + min_val
            start = _ * super_block_size + j * 32
            end = start + 32
            result[start : start + 16] = v0
            result[start + 16 : start + 32] = v1
            offset += 16
    return result[:n_elements]


def dequantize_q5_0(data: bytes, n_elements: int) -> np.ndarray:
    block_size = 32
    n_blocks = (n_elements + block_size - 1) // block_size
    scales = (
        np.frombuffer(data[: n_blocks * 2], dtype=np.float16).astype(np.float32).copy()
    )
    quants = np.frombuffer(
        data[n_blocks * 2 : n_blocks * 2 + n_blocks * 20], dtype=np.uint8
    ).copy()
    result = np.zeros(n_elements, dtype=np.float32)
    for i in range(n_blocks):
        scale = scales[i]
        qs = quants[i * 20 : (i + 1) * 20]
        q0 = (qs[:16] >> 4) & 0x0F
        q1 = qs[:16] & 0x0F
        qh = (qs[16:] >> 0) & 0x1F
        q0 = np.where(q0 >= 8, q0 - 16, q0).astype(np.float32)
        q1 = np.where(q1 >= 8, q1 - 16, q1).astype(np.float32)
        result[i * 32 + 0 : i * 32 + 16] = scale * (
            q0 + (qh.astype(np.float32) - 16) * (1 << 4)
        )
        result[i * 32 + 16 : i * 32 + 32] = scale * (
            q1 + ((qh >> 5).astype(np.float32) - 16) * (1 << 4)
        )
    return result[:n_elements]


def dequantize_q5_k(data: bytes, n_elements: int) -> np.ndarray:
    super_block_size = 256
    block_size = 32
    n_super_blocks = (n_elements + super_block_size - 1) // super_block_size
    result = np.zeros(n_elements, dtype=np.float32)
    offset = 0
    for _ in range(n_super_blocks):
        if offset >= len(data):
            break
        scales = np.frombuffer(data[offset : offset + 16], dtype=np.float16).astype(
            np.float32
        )
        mins = np.frombuffer(data[offset + 16 : offset + 32], dtype=np.float16).astype(
            np.float32
        )
        offset += 32
        for j in range(8):
            scale = scales[j]
            min_val = mins[j]
            q = np.frombuffer(data[offset : offset + 20], dtype=np.uint8)
            q0 = (q[:16] >> 4) & 0x0F
            q1 = q[:16] & 0x0F
            qh = (q[16:] >> 0) & 0x1F
            v0 = (
                np.where(q0 >= 8, q0 - 16, q0) + (qh.astype(np.float32) - 16) * 16
            ) * scale + min_val
            v1 = (
                np.where(q1 >= 8, q1 - 16, q1)
                + ((qh >> 5).astype(np.float32) - 16) * 16
            ) * scale + min_val
            start = _ * super_block_size + j * 32
            result[start : start + 16] = v0
            result[start + 16 : start + 32] = v1
            offset += 20
    return result[:n_elements]


def dequantize_q6_k(data: bytes, n_elements: int) -> np.ndarray:
    super_block_size = 256
    block_size = 32
    n_super_blocks = (n_elements + super_block_size - 1) // super_block_size
    result = np.zeros(n_elements, dtype=np.float32)
    offset = 0
    for _ in range(n_super_blocks):
        if offset >= len(data):
            break
        scales = np.frombuffer(data[offset : offset + 24], dtype=np.float16).astype(
            np.float32
        )
        offset += 24
        for j in range(8):
            scale = scales[j]
            q = np.frombuffer(data[offset : offset + 24], dtype=np.uint8)
            q0 = (q[:16] >> 4) & 0x0F
            q1 = q[:16] & 0x0F
            qh = (q[16:20] >> 0) & 0x3F
            ql = (q[20:24] >> 0) & 0x3F
            v0 = (
                np.where(q0 >= 8, q0 - 16, q0) + (qh.astype(np.float32) - 32) * 16
            ) * scale
            v1 = (
                np.where(q1 >= 8, q1 - 16, q1) + (ql.astype(np.float32) - 32) * 16
            ) * scale
            start = _ * super_block_size + j * 32
            result[start : start + 16] = v0
            result[start + 16 : start + 32] = v1
            offset += 24
    return result[:n_elements]


def dequantize_q2_k(data: bytes, n_elements: int) -> np.ndarray:
    super_block_size = 256
    block_size = 32
    n_super_blocks = (n_elements + super_block_size - 1) // super_block_size
    result = np.zeros(n_elements, dtype=np.float32)
    offset = 0
    for _ in range(n_super_blocks):
        if offset >= len(data):
            break
        d = np.frombuffer(data[offset : offset + 2], dtype=np.float16)[0]
        scales = np.frombuffer(data[offset + 2 : offset + 18], dtype=np.uint8)
        mins = (scales >> 6) & 3
        scales = scales & 63
        offset += 18
        for j in range(8):
            scale = (scales[j] - 31) * d / 63
            min_val = (mins[j] - 2) * d / 63
            q = np.frombuffer(data[offset : offset + 12], dtype=np.uint8)
            q0 = (q[:8] >> 0) & 3
            q1 = (q[:8] >> 2) & 3
            q2 = (q[:8] >> 4) & 3
            q3 = (q[:8] >> 6) & 3
            bits = np.concatenate([q[8:12]]).astype(np.uint8)
            q4 = (bits >> 0) & 3
            q5 = (bits >> 2) & 3
            q6 = (bits >> 4) & 3
            q7 = (bits >> 6) & 3
            all_q = np.array([q0, q1, q2, q3, q4, q5, q6, q7]).T.flatten()
            v = (all_q.astype(np.float32) - 3) * scale + min_val
            start = _ * super_block_size + j * 32
            result[start : start + 32] = v[:32]
            offset += 12
    return result[:n_elements]


def dequantize_q3_k(data: bytes, n_elements: int) -> np.ndarray:
    super_block_size = 256
    block_size = 32
    n_super_blocks = (n_elements + super_block_size - 1) // super_block_size
    result = np.zeros(n_elements, dtype=np.float32)
    offset = 0
    for _ in range(n_super_blocks):
        if offset >= len(data):
            break
        scales = np.frombuffer(data[offset : offset + 16], dtype=np.uint8)
        offset += 16
        for j in range(8):
            scale = (scales[j] - 31) / 63
            q = np.frombuffer(data[offset : offset + 12], dtype=np.uint8)
            q0 = (q[:12] >> 0) & 7
            q1 = (q[:12] >> 3) & 7
            bits = np.concatenate([q[12:]]).astype(np.uint8)
            q2 = (bits >> 0) & 7
            q3 = (bits >> 3) & 7
            all_q = np.array([q0, q1, q2, q3]).T.flatten()
            v = (all_q.astype(np.float32) - 4) * scale
            start = _ * super_block_size + j * 32
            result[start : start + 32] = v[:32]
            offset += 12
    return result[:n_elements]


def dequantize_iq4_xs(data: bytes, n_elements: int) -> np.ndarray:
    block_size = 32
    n_blocks = (n_elements + block_size - 1) // block_size
    result = np.zeros(n_elements, dtype=np.float32)
    offset = 0
    for i in range(n_blocks):
        if offset + 2 > len(data):
            break
        d = np.frombuffer(data[offset : offset + 2], dtype=np.float16)[0]
        offset += 2
        q = np.frombuffer(data[offset : offset + 16], dtype=np.uint8)
        offset += 16
        qs = np.zeros(32, dtype=np.float32)
        for q_idx in range(16):
            b0 = (q[q_idx] >> 0) & 0xF
            b1 = (q[q_idx] >> 4) & 0xF
            qs[q_idx * 2] = b0 - 8 if b0 < 8 else b0 - 16
            qs[q_idx * 2 + 1] = b1 - 8 if b1 < 8 else b1 - 16
        result[i * 32 : (i + 1) * 32] = qs * d
    return result[:n_elements]


def resolve_decode_kind(tensor: TensorInfo, decode_as: str) -> str:
    from .constants import EDITABLE_TYPE_NAMES

    if decode_as and decode_as != "auto":
        return decode_as.upper()
    if tensor.editable_kind in {"F32", "F16", "BF16"}:
        return tensor.editable_kind
    if tensor.editable_kind == "F16_OR_BF16":
        return "F16"
    return tensor.editable_kind


def decode_tensor(path: str, tensor: TensorInfo, decode_as: str = "auto") -> np.ndarray:
    kind = resolve_decode_kind(tensor, decode_as)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            raw = mm[tensor.abs_offset : tensor.abs_offset + tensor.n_bytes]
        finally:
            mm.close()
    if kind == "F32":
        arr = np.frombuffer(raw, dtype="<f4").copy()
    elif kind == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif kind == "BF16":
        arr = bf16_to_f32(np.frombuffer(raw, dtype="<u2"))
    elif kind == "Q8_0":
        arr = dequantize_q8_0(raw, tensor.n_elements)
    elif kind == "Q4_0":
        arr = dequantize_q4_0(raw, tensor.n_elements)
    elif kind == "Q4_1":
        arr = dequantize_q4_0(raw, tensor.n_elements)
    elif kind == "Q5_0":
        arr = dequantize_q5_0(raw, tensor.n_elements)
    elif kind == "Q5_1":
        arr = dequantize_q5_0(raw, tensor.n_elements)
    elif kind == "Q2_K":
        arr = dequantize_q2_k(raw, tensor.n_elements)
    elif kind == "Q3_K":
        arr = dequantize_q3_k(raw, tensor.n_elements)
    elif kind == "Q4_K":
        arr = dequantize_q4_k(raw, tensor.n_elements)
    elif kind == "Q5_K":
        arr = dequantize_q5_k(raw, tensor.n_elements)
    elif kind == "Q6_K":
        arr = dequantize_q6_k(raw, tensor.n_elements)
    elif kind == "Q8_K":
        arr = dequantize_q8_0(raw, tensor.n_elements)
    elif kind == "IQ4_XS":
        arr = dequantize_iq4_xs(raw, tensor.n_elements)
    else:
        raise ValueError(f"Tensor '{tensor.name}' is not editable ({kind}).")
    return arr.reshape(tensor.shape) if tensor.shape else arr


def encode_tensor(
    arr_f32: np.ndarray, tensor: TensorInfo, decode_as: str = "auto"
) -> bytes:
    kind = resolve_decode_kind(tensor, decode_as)
    arr_f32 = np.asarray(arr_f32, dtype=np.float32)
    if kind == "F32":
        data = arr_f32.astype("<f4", copy=False).tobytes(order="C")
    elif kind == "F16":
        data = arr_f32.astype("<f2").tobytes(order="C")
    elif kind == "BF16":
        data = f32_to_bf16(arr_f32).astype("<u2", copy=False).tobytes(order="C")
    else:
        raise ValueError(f"Tensor '{tensor.name}' cannot be written as {kind}.")
    if len(data) != tensor.n_bytes:
        raise ValueError(
            f"Encoded byte length mismatch for {tensor.name}: {len(data)} != {tensor.n_bytes}"
        )
    return data


def transform_array(
    arr: np.ndarray,
    scale: float,
    bias: float,
    clip_min: Optional[float],
    clip_max: Optional[float],
) -> np.ndarray:
    import math

    out = arr.astype(np.float32) * np.float32(scale) + np.float32(bias)
    lo = (
        None
        if clip_min is None or (isinstance(clip_min, float) and math.isnan(clip_min))
        else clip_min
    )
    hi = (
        None
        if clip_max is None or (isinstance(clip_max, float) and math.isnan(clip_max))
        else clip_max
    )
    if lo is not None or hi is not None:
        out = np.clip(out, -np.inf if lo is None else lo, np.inf if hi is None else hi)
    return out


def build_transform_preview(
    arr_before: np.ndarray, arr_after: np.ndarray, preview_n: int = 16
) -> Tuple[str, Any]:
    import pandas as pd

    flat_before = arr_before.reshape(-1)
    flat_after = arr_after.reshape(-1)
    delta = flat_after - flat_before
    n = min(preview_n, flat_before.shape[0])
    preview = pd.DataFrame(
        [
            {
                "flat_index": i,
                "before": float(flat_before[i]),
                "after": float(flat_after[i]),
                "delta": float(delta[i]),
            }
            for i in range(n)
        ]
    )
    text = (
        "### Before / After Preview\n"
        f"- Mean: **{float(flat_before.mean()):.6g} → {float(flat_after.mean()):.6g}**\n"
        f"- Std: **{float(flat_before.std()):.6g} → {float(flat_after.std()):.6g}**\n"
        f"- Min / Max: **{float(flat_before.min()):.6g}/{float(flat_before.max()):.6g} → {float(flat_after.min()):.6g}/{float(flat_after.max()):.6g}**\n"
        f"- Mean absolute delta: **{float(np.abs(delta).mean()):.6g}**\n"
        f"- Max absolute delta: **{float(np.abs(delta).max()):.6g}**"
    )
    return text, preview


def parse_indices(indices_text: str, shape: Tuple[int, ...]) -> Tuple[int, ...]:
    parts = [p.strip() for p in indices_text.split(",") if p.strip()]
    if len(parts) != len(shape):
        raise ValueError(
            f"Expected {len(shape)} indices for shape {list(shape)}, got {len(parts)}"
        )
    idxs = tuple(int(p) for p in parts)
    for i, dim in zip(idxs, shape):
        if i < 0 or i >= dim:
            raise ValueError(f"Index {idxs} is out of bounds for shape {list(shape)}")
    return idxs


def parse_slice_spec(
    arr: np.ndarray, axis: int, index: int
) -> Tuple[np.ndarray, Tuple[Any, ...]]:
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(f"Axis {axis} out of bounds for tensor ndim {arr.ndim}")
    if index < 0 or index >= arr.shape[axis]:
        raise ValueError(
            f"Index {index} out of bounds for axis {axis} with size {arr.shape[axis]}"
        )
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = index
    slicer_tuple = tuple(slicer)
    return arr[slicer_tuple], slicer_tuple
