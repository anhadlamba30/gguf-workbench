from __future__ import annotations

import mmap
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from .constants import GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16

from .parser import TensorInfo


def bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    arr_u32 = arr_u16.astype(np.uint32) << 16
    return arr_u32.view(np.float32)


def f32_to_bf16(arr_f32: np.ndarray) -> np.ndarray:
    arr_u32 = np.asarray(arr_f32, dtype=np.float32).view(np.uint32)
    rounding_bias = ((arr_u32 >> 16) & 1) + 0x7FFF
    return ((arr_u32 + rounding_bias) >> 16).astype(np.uint16)


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
