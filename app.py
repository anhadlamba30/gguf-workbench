#!/usr/bin/env python3
from __future__ import annotations

import math
import mmap
import os
import shutil
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
import pandas as pd

APP_TITLE = "GGUF Workbench V2.1"
DEFAULT_ALIGNMENT = 32
GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = {2, 3}

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# Known ggml tensor types used by this build
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30
GGML_TYPE_NAMES = {
    0: "F32",
    1: "F16",
    30: "BF16",
}

EDITABLE_TYPE_NAMES = {"F32", "F16", "BF16", "F16_OR_BF16"}


def align_offset(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


class GGUFParseError(RuntimeError):
    pass


class BinaryReader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def tell(self) -> int:
        return self.pos

    def read(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.data):
            raise GGUFParseError(f"Unexpected EOF while reading {n} bytes at offset {self.pos}")
        out = self.data[self.pos:end]
        self.pos = end
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.read(1))[0]

    def u8(self) -> int:
        return struct.unpack("<B", self.read(1))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.read(2))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read(2))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read(8))[0]

    def boolean(self) -> bool:
        return bool(self.read(1)[0])

    def string(self) -> str:
        n = self.u64()
        raw = self.read(n)
        return raw.decode("utf-8", errors="replace")


@dataclass
class TensorInfo:
    name: str
    shape: Tuple[int, ...]
    stored_dims: Tuple[int, ...]
    ggml_type: int
    ggml_type_name: str
    offset: int
    abs_offset: int
    n_elements: int
    n_bytes: int
    editable_kind: str

    def to_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "elements": self.n_elements,
            "type_id": self.ggml_type,
            "type": self.ggml_type_name,
            "bytes": self.n_bytes,
            "offset": self.abs_offset,
            "editable": self.editable_kind in EDITABLE_TYPE_NAMES,
            "edit_kind": self.editable_kind,
        }


@dataclass
class GGUFManifest:
    path: str
    version: int
    n_tensors: int
    n_kv: int
    alignment: int
    metadata: Dict[str, Any]
    tensors: List[TensorInfo]
    tensor_data_start: int
    file_size: int

    def tensor_map(self) -> Dict[str, TensorInfo]:
        return {t.name: t for t in self.tensors}

    def editable_tensors(self) -> List[str]:
        return [t.name for t in self.tensors if t.editable_kind in EDITABLE_TYPE_NAMES]


@dataclass
class BatchOperation:
    """Represents a single operation to be added to the batch queue."""
    op_type: str  # "scalar", "transform", "slice"
    tensor_name: str
    decode_as: str
    parameters: Dict[str, Any]
    
    def display_label(self) -> str:
        """Generate a user-friendly label for this operation."""
        if self.op_type == "scalar":
            idx = self.parameters.get("indices", "?")
            val = self.parameters.get("new_value", "?")
            return f"Scalar: {self.tensor_name}[{idx}] → {val:.6g}"
        elif self.op_type == "transform":
            scale = self.parameters.get("scale", 1.0)
            bias = self.parameters.get("bias", 0.0)
            return f"Transform: {self.tensor_name} × {scale:.4g} + {bias:.4g}"
        elif self.op_type == "slice":
            axis = self.parameters.get("axis", 0)
            index = self.parameters.get("index", 0)
            mode = self.parameters.get("mode", "")
            return f"Slice: {self.tensor_name}[axis={axis}, idx={index}] ({mode})"
        return f"Unknown: {self.tensor_name}"


class GGUFParser:
    def __init__(self, file_path: str | os.PathLike[str]):
        self.path = Path(file_path)
        self._data = self.path.read_bytes()
        self.reader = BinaryReader(self._data)

    def parse(self) -> GGUFManifest:
        r = self.reader
        magic = r.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFParseError(f"Not a GGUF file: magic={magic!r}")
        version = r.u32()
        if version not in SUPPORTED_VERSIONS:
            raise GGUFParseError(f"Unsupported GGUF version {version}. Supported: {sorted(SUPPORTED_VERSIONS)}")
        n_tensors = r.u64()
        n_kv = r.u64()

        metadata: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            value_type = r.u32()
            metadata[key] = self._read_value(r, value_type)

        alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT) or DEFAULT_ALIGNMENT)
        tensor_descs: List[Dict[str, Any]] = []
        for _ in range(n_tensors):
            name = r.string()
            n_dimensions = r.u32()
            dims_stored = tuple(r.u64() for _ in range(n_dimensions))
            ggml_type = r.u32()
            offset = r.u64()
            tensor_descs.append({
                "name": name,
                "dims_stored": dims_stored,
                "shape": tuple(reversed(dims_stored)),
                "ggml_type": ggml_type,
                "offset": offset,
            })

        tensor_data_start = align_offset(r.tell(), alignment)
        file_size = len(self._data)
        tensors: List[TensorInfo] = []
        for desc in tensor_descs:
            abs_off = tensor_data_start + int(desc["offset"])
            n_elements = int(np.prod(desc["shape"], dtype=np.int64)) if desc["shape"] else 1
            tensors.append(
                TensorInfo(
                    name=desc["name"],
                    shape=tuple(int(x) for x in desc["shape"]),
                    stored_dims=tuple(int(x) for x in desc["dims_stored"]),
                    ggml_type=int(desc["ggml_type"]),
                    ggml_type_name=GGML_TYPE_NAMES.get(int(desc["ggml_type"]), f"TYPE_{desc['ggml_type']}"),
                    offset=int(desc["offset"]),
                    abs_offset=int(abs_off),
                    n_elements=n_elements,
                    n_bytes=0,
                    editable_kind="UNKNOWN",
                )
            )

        tensors_sorted = sorted(tensors, key=lambda t: t.abs_offset)
        for i, tensor in enumerate(tensors_sorted):
            next_abs = tensors_sorted[i + 1].abs_offset if i + 1 < len(tensors_sorted) else file_size
            tensor.n_bytes = int(next_abs - tensor.abs_offset)
            tensor.editable_kind = self._infer_editable_kind(tensor)
            if tensor.abs_offset > file_size or tensor.n_bytes < 0:
                raise GGUFParseError(f"Invalid tensor bounds for {tensor.name}")

        tensor_map = {t.name: t for t in tensors_sorted}
        tensors = [tensor_map[d["name"]] for d in tensor_descs]
        return GGUFManifest(
            path=str(self.path),
            version=int(version),
            n_tensors=int(n_tensors),
            n_kv=int(n_kv),
            alignment=alignment,
            metadata=metadata,
            tensors=tensors,
            tensor_data_start=tensor_data_start,
            file_size=file_size,
        )

    def _read_value(self, r: BinaryReader, value_type: int) -> Any:
        if value_type == GGUF_TYPE_UINT8:
            return r.u8()
        if value_type == GGUF_TYPE_INT8:
            return r.i8()
        if value_type == GGUF_TYPE_UINT16:
            return r.u16()
        if value_type == GGUF_TYPE_INT16:
            return r.i16()
        if value_type == GGUF_TYPE_UINT32:
            return r.u32()
        if value_type == GGUF_TYPE_INT32:
            return r.i32()
        if value_type == GGUF_TYPE_FLOAT32:
            return r.f32()
        if value_type == GGUF_TYPE_BOOL:
            return r.boolean()
        if value_type == GGUF_TYPE_STRING:
            return r.string()
        if value_type == GGUF_TYPE_ARRAY:
            item_type = r.u32()
            n = r.u64()
            return [self._read_value(r, item_type) for _ in range(n)]
        if value_type == GGUF_TYPE_UINT64:
            return r.u64()
        if value_type == GGUF_TYPE_INT64:
            return r.i64()
        if value_type == GGUF_TYPE_FLOAT64:
            return r.f64()
        raise GGUFParseError(f"Unsupported metadata value type {value_type}")

    def _infer_editable_kind(self, tensor: TensorInfo) -> str:
        if tensor.ggml_type == GGML_TYPE_F32 and tensor.n_bytes == tensor.n_elements * 4:
            return "F32"
        if tensor.ggml_type == GGML_TYPE_F16 and tensor.n_bytes == tensor.n_elements * 2:
            return "F16"
        if tensor.ggml_type == GGML_TYPE_BF16 and tensor.n_bytes == tensor.n_elements * 2:
            return "BF16"
        if tensor.n_bytes == tensor.n_elements * 4:
            return "F32"
        if tensor.n_bytes == tensor.n_elements * 2:
            return "F16_OR_BF16"
        return GGML_TYPE_NAMES.get(tensor.ggml_type, f"TYPE_{tensor.ggml_type}")


def bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    arr_u32 = arr_u16.astype(np.uint32) << 16
    return arr_u32.view(np.float32)


def f32_to_bf16(arr_f32: np.ndarray) -> np.ndarray:
    arr_u32 = np.asarray(arr_f32, dtype=np.float32).view(np.uint32)
    rounding_bias = ((arr_u32 >> 16) & 1) + 0x7FFF
    return ((arr_u32 + rounding_bias) >> 16).astype(np.uint16)


def resolve_decode_kind(tensor: TensorInfo, decode_as: str) -> str:
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
            raw = mm[tensor.abs_offset:tensor.abs_offset + tensor.n_bytes]
        finally:
            mm.close()
    if kind == "F32":
        arr = np.frombuffer(raw, dtype="<f4").copy()
    elif kind == "F16":
        arr = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif kind == "BF16":
        arr = bf16_to_f32(np.frombuffer(raw, dtype="<u2"))
    else:
        raise gr.Error(f"Tensor '{tensor.name}' is not editable in v2.1 ({kind}).")
    return arr.reshape(tensor.shape) if tensor.shape else arr


def encode_tensor(arr_f32: np.ndarray, tensor: TensorInfo, decode_as: str = "auto") -> bytes:
    kind = resolve_decode_kind(tensor, decode_as)
    arr_f32 = np.asarray(arr_f32, dtype=np.float32)
    if kind == "F32":
        data = arr_f32.astype("<f4", copy=False).tobytes(order="C")
    elif kind == "F16":
        data = arr_f32.astype("<f2").tobytes(order="C")
    elif kind == "BF16":
        data = f32_to_bf16(arr_f32).astype("<u2", copy=False).tobytes(order="C")
    else:
        raise gr.Error(f"Tensor '{tensor.name}' cannot be written as {kind}.")
    if len(data) != tensor.n_bytes:
        raise gr.Error(f"Encoded byte length mismatch for {tensor.name}: {len(data)} != {tensor.n_bytes}")
    return data


def manifest_to_dict(manifest: GGUFManifest) -> Dict[str, Any]:
    return {
        "path": manifest.path,
        "version": manifest.version,
        "n_tensors": manifest.n_tensors,
        "n_kv": manifest.n_kv,
        "alignment": manifest.alignment,
        "metadata": manifest.metadata,
        "tensors": [asdict(t) for t in manifest.tensors],
        "tensor_data_start": manifest.tensor_data_start,
        "file_size": manifest.file_size,
    }


def manifest_from_dict(data: Dict[str, Any]) -> GGUFManifest:
    if not data:
        raise gr.Error("No GGUF loaded yet.")
    tensors = [TensorInfo(**t) for t in data["tensors"]]
    return GGUFManifest(
        path=data["path"],
        version=data["version"],
        n_tensors=data["n_tensors"],
        n_kv=data["n_kv"],
        alignment=data["alignment"],
        metadata=data["metadata"],
        tensors=tensors,
        tensor_data_start=data["tensor_data_start"],
        file_size=data["file_size"],
    )


def manifest_summary(manifest: GGUFManifest) -> str:
    editable = sum(1 for t in manifest.tensors if t.editable_kind in EDITABLE_TYPE_NAMES)
    return (
        f"### {Path(manifest.path).name}\n"
        f"- Version: **{manifest.version}**\n"
        f"- File size: **{manifest.file_size / (1024 ** 3):.4f} GiB**\n"
        f"- Metadata entries: **{manifest.n_kv}**\n"
        f"- Tensors: **{manifest.n_tensors}**\n"
        f"- Alignment: **{manifest.alignment}**\n"
        f"- Editable tensors (float-like): **{editable}**"
    )


def load_manifest(file_path: str):
    parser = GGUFParser(file_path)
    manifest = parser.parse()
    meta_rows = []
    for k, v in manifest.metadata.items():
        preview = v
        if isinstance(v, list) and len(v) > 16:
            preview = v[:16] + [f"… ({len(v)} total)"]
        meta_rows.append({"key": k, "python_type": type(v).__name__, "value": preview})
    tensor_rows = [t.to_row() for t in manifest.tensors]
    meta_df = pd.DataFrame(meta_rows)
    tensor_df = pd.DataFrame(tensor_rows)
    choices = manifest.editable_tensors()
    return manifest, meta_df, tensor_df, choices


def default_output_path(input_path: str, suffix: str = ".patched.gguf") -> str:
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}{suffix}"))


def filter_tensor_table(manifest_dict: Dict[str, Any], query: str, editable_only: bool):
    manifest = manifest_from_dict(manifest_dict)
    rows = [t.to_row() for t in manifest.tensors]
    df = pd.DataFrame(rows)
    if editable_only:
        df = df[df["editable"] == True]
    q = (query or "").strip().lower()
    if q:
        df = df[df["name"].str.lower().str.contains(q, regex=False)]
    return df.reset_index(drop=True)


def inspect_tensor(manifest_dict: Dict[str, Any], tensor_name: str, decode_as: str, max_items: int):
    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map().get(tensor_name)
    if tensor is None:
        raise gr.Error(f"Tensor not found: {tensor_name}")
    arr = decode_tensor(manifest.path, tensor, decode_as)
    flat = arr.reshape(-1)
    n = min(int(max_items), flat.shape[0])
    preview = pd.DataFrame([{"flat_index": i, "value": float(v)} for i, v in enumerate(flat[:n])])
    text = (
        f"### Tensor: `{tensor.name}`\n"
        f"- Shape: **{list(tensor.shape)}**\n"
        f"- Stored dims: **{list(tensor.stored_dims)}**\n"
        f"- Type ID: **{tensor.ggml_type}**\n"
        f"- Type guess: **{tensor.editable_kind}**\n"
        f"- Decode as: **{resolve_decode_kind(tensor, decode_as)}**\n"
        f"- Elements: **{tensor.n_elements}**\n"
        f"- Min / Max: **{float(flat.min()):.6g} / {float(flat.max()):.6g}**\n"
        f"- Mean / Std: **{float(flat.mean()):.6g} / {float(flat.std()):.6g}**"
    )
    return text, preview


def parse_indices(indices_text: str, shape: Sequence[int]) -> Tuple[int, ...]:
    parts = [p.strip() for p in indices_text.split(",") if p.strip()]
    if len(parts) != len(shape):
        raise gr.Error(f"Expected {len(shape)} indices for shape {list(shape)}, got {len(parts)}")
    idxs = tuple(int(p) for p in parts)
    for i, dim in zip(idxs, shape):
        if i < 0 or i >= dim:
            raise gr.Error(f"Index {idxs} is out of bounds for shape {list(shape)}")
    return idxs


def write_tensor_patch(manifest: GGUFManifest, tensor: TensorInfo, arr: np.ndarray, decode_as: str, output_path: str):
    payload = encode_tensor(arr, tensor, decode_as)
    shutil.copyfile(manifest.path, output_path)
    with open(output_path, "r+b") as f:
        f.seek(tensor.abs_offset)
        f.write(payload)


def patch_scalar(manifest_dict: Dict[str, Any], tensor_name: str, decode_as: str, indices_text: str, new_value: float, output_path: str):
    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map()[tensor_name]
    output_path = output_path or default_output_path(manifest.path)
    arr = decode_tensor(manifest.path, tensor, decode_as)
    idxs = parse_indices(indices_text, arr.shape)
    before = float(arr[idxs])
    arr[idxs] = np.float32(new_value)
    write_tensor_patch(manifest, tensor, arr, decode_as, output_path)
    return f"Patched `{tensor.name}{idxs}` from **{before:.8g}** to **{float(new_value):.8g}**\n\nSaved: `{output_path}`"


def build_transform_preview(arr_before: np.ndarray, arr_after: np.ndarray, preview_n: int = 16) -> Tuple[str, pd.DataFrame]:
    flat_before = arr_before.reshape(-1)
    flat_after = arr_after.reshape(-1)
    delta = flat_after - flat_before
    n = min(preview_n, flat_before.shape[0])
    preview = pd.DataFrame([
        {
            "flat_index": i,
            "before": float(flat_before[i]),
            "after": float(flat_after[i]),
            "delta": float(delta[i]),
        }
        for i in range(n)
    ])
    text = (
        "### Before / After Preview\n"
        f"- Mean: **{float(flat_before.mean()):.6g} → {float(flat_after.mean()):.6g}**\n"
        f"- Std: **{float(flat_before.std()):.6g} → {float(flat_after.std()):.6g}**\n"
        f"- Min / Max: **{float(flat_before.min()):.6g}/{float(flat_before.max()):.6g} → {float(flat_after.min()):.6g}/{float(flat_after.max()):.6g}**\n"
        f"- Mean absolute delta: **{float(np.abs(delta).mean()):.6g}**\n"
        f"- Max absolute delta: **{float(np.abs(delta).max()):.6g}**"
    )
    return text, preview


def transform_array(arr: np.ndarray, scale: float, bias: float, clip_min: Optional[float], clip_max: Optional[float]) -> np.ndarray:
    out = arr.astype(np.float32) * np.float32(scale) + np.float32(bias)
    lo = None if clip_min is None or (isinstance(clip_min, float) and math.isnan(clip_min)) else clip_min
    hi = None if clip_max is None or (isinstance(clip_max, float) and math.isnan(clip_max)) else clip_max
    if lo is not None or hi is not None:
        out = np.clip(out, -np.inf if lo is None else lo, np.inf if hi is None else hi)
    return out


def preview_transform(manifest_dict: Dict[str, Any], tensor_name: str, decode_as: str, scale: float, bias: float, clip_min: Optional[float], clip_max: Optional[float]):
    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map()[tensor_name]
    before = decode_tensor(manifest.path, tensor, decode_as)
    after = transform_array(before, scale, bias, clip_min, clip_max)
    return build_transform_preview(before, after)


def patch_transform(manifest_dict: Dict[str, Any], tensor_name: str, decode_as: str, scale: float, bias: float, clip_min: Optional[float], clip_max: Optional[float], output_path: str):
    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map()[tensor_name]
    output_path = output_path or default_output_path(manifest.path, ".transformed.gguf")
    before = decode_tensor(manifest.path, tensor, decode_as)
    after = transform_array(before, scale, bias, clip_min, clip_max)
    write_tensor_patch(manifest, tensor, after, decode_as, output_path)
    preview_text, _ = build_transform_preview(before, after)
    return preview_text + f"\n\nSaved: `{output_path}`"


def parse_slice_spec(arr: np.ndarray, axis: int, index: int) -> Tuple[np.ndarray, Tuple[Any, ...]]:
    if axis < 0 or axis >= arr.ndim:
        raise gr.Error(f"Axis {axis} out of bounds for tensor ndim {arr.ndim}")
    if index < 0 or index >= arr.shape[axis]:
        raise gr.Error(f"Index {index} out of bounds for axis {axis} with size {arr.shape[axis]}")
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = index
    slicer_tuple = tuple(slicer)
    return arr[slicer_tuple], slicer_tuple


def preview_slice_edit(manifest_dict: Dict[str, Any], tensor_name: str, decode_as: str, axis: int, index: int, mode: str, value: float, scale: float, bias: float):
    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map()[tensor_name]
    arr = decode_tensor(manifest.path, tensor, decode_as)
    slice_view, _ = parse_slice_spec(arr, int(axis), int(index))
    before = np.array(slice_view, copy=True)
    after = before.copy()
    if mode == "set_constant":
        after[...] = np.float32(value)
    else:
        after[...] = after * np.float32(scale) + np.float32(bias)
    preview_text, preview_df = build_transform_preview(before, after)
    preview_text = f"### Slice Preview (axis={axis}, index={index})\n\n" + preview_text
    return preview_text, preview_df


def patch_slice(manifest_dict: Dict[str, Any], tensor_name: str, decode_as: str, axis: int, index: int, mode: str, value: float, scale: float, bias: float, output_path: str):
    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map()[tensor_name]
    output_path = output_path or default_output_path(manifest.path, ".slice.gguf")
    arr = decode_tensor(manifest.path, tensor, decode_as)
    slice_view, slicer = parse_slice_spec(arr, int(axis), int(index))
    before = np.array(slice_view, copy=True)
    if mode == "set_constant":
        arr[slicer] = np.float32(value)
    else:
        arr[slicer] = slice_view * np.float32(scale) + np.float32(bias)
    after = np.array(arr[slicer], copy=True)
    write_tensor_patch(manifest, tensor, arr, decode_as, output_path)
    preview_text, _ = build_transform_preview(before, after)
    return f"### Slice Patch Applied (axis={axis}, index={index})\n\n" + preview_text + f"\n\nSaved: `{output_path}`"


# ==================== BATCH OPERATIONS ====================

def batch_op_to_dict(op: BatchOperation) -> Dict[str, Any]:
    """Serialize a batch operation."""
    return {
        "op_type": op.op_type,
        "tensor_name": op.tensor_name,
        "decode_as": op.decode_as,
        "parameters": op.parameters,
    }


def batch_op_from_dict(data: Dict[str, Any]) -> BatchOperation:
    """Deserialize a batch operation."""
    return BatchOperation(
        op_type=data["op_type"],
        tensor_name=data["tensor_name"],
        decode_as=data["decode_as"],
        parameters=data["parameters"],
    )


def batch_add_scalar(batch_list: List[Dict[str, Any]], tensor_name: str, decode_as: str, indices_text: str, new_value: float) -> Tuple[List[Dict[str, Any]], str]:
    """Add a scalar patch operation to the batch."""
    if not tensor_name:
        raise gr.Error("Please select a tensor")
    op = BatchOperation(
        op_type="scalar",
        tensor_name=tensor_name,
        decode_as=decode_as,
        parameters={"indices": indices_text, "new_value": new_value},
    )
    batch_list.append(batch_op_to_dict(op))
    msg = f"✓ Added: {op.display_label()}"
    return batch_list, msg


def batch_add_transform(batch_list: List[Dict[str, Any]], tensor_name: str, decode_as: str, scale: float, bias: float, clip_min: Optional[float], clip_max: Optional[float]) -> Tuple[List[Dict[str, Any]], str]:
    """Add a transform operation to the batch."""
    if not tensor_name:
        raise gr.Error("Please select a tensor")
    op = BatchOperation(
        op_type="transform",
        tensor_name=tensor_name,
        decode_as=decode_as,
        parameters={"scale": scale, "bias": bias, "clip_min": clip_min, "clip_max": clip_max},
    )
    batch_list.append(batch_op_to_dict(op))
    msg = f"✓ Added: {op.display_label()}"
    return batch_list, msg


def batch_add_slice(batch_list: List[Dict[str, Any]], tensor_name: str, decode_as: str, axis: int, index: int, mode: str, value: float, scale: float, bias: float) -> Tuple[List[Dict[str, Any]], str]:
    """Add a slice patch operation to the batch."""
    if not tensor_name:
        raise gr.Error("Please select a tensor")
    op = BatchOperation(
        op_type="slice",
        tensor_name=tensor_name,
        decode_as=decode_as,
        parameters={"axis": axis, "index": index, "mode": mode, "value": value, "scale": scale, "bias": bias},
    )
    batch_list.append(batch_op_to_dict(op))
    msg = f"✓ Added: {op.display_label()}"
    return batch_list, msg


def render_batch_queue(batch_list: List[Dict[str, Any]]) -> str:
    """Render the batch queue as markdown."""
    if not batch_list:
        return "### Batch Queue\nEmpty. Add operations above to get started."
    
    lines = ["### Batch Queue", f"**{len(batch_list)} operation(s) pending:**\n"]
    for i, op_dict in enumerate(batch_list, 1):
        op = batch_op_from_dict(op_dict)
        lines.append(f"{i}. {op.display_label()}")
    return "\n".join(lines)


def apply_batch(batch_list: List[Dict[str, Any]], manifest_dict: Dict[str, Any], output_path: str) -> str:
    """Apply all queued batch operations to a copy of the original GGUF."""
    if not batch_list:
        raise gr.Error("Batch is empty. Add operations first.")
    if not output_path:
        raise gr.Error("Please specify an output path.")
    
    manifest = manifest_from_dict(manifest_dict)
    
    # Start fresh from the original file
    shutil.copyfile(manifest.path, output_path)
    
    # Apply each operation to the copied file
    for i, op_dict in enumerate(batch_list, 1):
        op = batch_op_from_dict(op_dict)
        tensor = manifest.tensor_map().get(op.tensor_name)
        if tensor is None:
            raise gr.Error(f"Operation {i}: Tensor '{op.tensor_name}' not found")
        
        # Read current state from the output file
        arr = decode_tensor(output_path, tensor, op.decode_as)
        
        try:
            if op.op_type == "scalar":
                indices = parse_indices(op.parameters["indices"], arr.shape)
                arr[indices] = np.float32(op.parameters["new_value"])
            elif op.op_type == "transform":
                arr = transform_array(
                    arr,
                    op.parameters["scale"],
                    op.parameters["bias"],
                    op.parameters.get("clip_min"),
                    op.parameters.get("clip_max"),
                )
            elif op.op_type == "slice":
                slice_view, slicer = parse_slice_spec(arr, int(op.parameters["axis"]), int(op.parameters["index"]))
                if op.parameters["mode"] == "set_constant":
                    arr[slicer] = np.float32(op.parameters["value"])
                else:
                    arr[slicer] = slice_view * np.float32(op.parameters["scale"]) + np.float32(op.parameters["bias"])
            else:
                raise gr.Error(f"Operation {i}: Unknown operation type {op.op_type}")
            
            # Write the modified array back
            write_tensor_patch(manifest, tensor, arr, op.decode_as, output_path)
        except Exception as e:
            raise gr.Error(f"Operation {i} ({op.op_type} on {op.tensor_name}) failed: {str(e)}")
    
    return f"✓ **Batch applied successfully!**\n\nApplied {len(batch_list)} operation(s).\n\nSaved to: `{output_path}`"


def clear_batch(batch_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """Clear all pending batch operations."""
    return [], "### Batch Queue\nEmpty. Add operations above to get started."


def on_load(file_path: str):
    if not file_path:
        raise gr.Error("Please provide a local GGUF path.")
    manifest, meta_df, tensor_df, choices = load_manifest(file_path)
    manifest_dict = manifest_to_dict(manifest)
    summary = manifest_summary(manifest)
    default_tensor = choices[0] if choices else None
    scalar_out = default_output_path(file_path)
    transform_out = default_output_path(file_path, ".transformed.gguf")
    slice_out = default_output_path(file_path, ".slice.gguf")
    batch_out = default_output_path(file_path, ".batch.gguf")
    return (
        manifest_dict,
        summary,
        meta_df,
        tensor_df,
        gr.update(choices=choices, value=default_tensor),
        gr.update(choices=choices, value=default_tensor),
        gr.update(choices=choices, value=default_tensor),
        gr.update(choices=choices, value=default_tensor),
        scalar_out,
        transform_out,
        slice_out,
        batch_out,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE) as demo:
        manifest_state = gr.State({})
        batch_queue = gr.State([])
        gr.Markdown(
            "# GGUF Workbench V2.1\n"
            "Functionality-first GGUF editor for **float-like tensors only**.\n\n"
            "### New in v2.1\n"
            "- tensor search / filter\n"
            "- transform preview with before/after diff\n"
            "- slice patching by axis + index\n"
            "- more in-UI help so the math is obvious\n\n"
            "### NEW: Batch Operations\n"
            "Queue multiple edits and apply them all at once to generate one final modified GGUF. See the **Batch Manager** tab."
        )

        with gr.Row():
            gguf_path = gr.Textbox(label="Local GGUF path", placeholder="/absolute/path/to/model.gguf", scale=4)
            load_btn = gr.Button("Load GGUF", variant="primary", scale=1)

        summary_md = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Metadata + Tensors"):
                meta_df = gr.Dataframe(label="Metadata")
                with gr.Row():
                    filter_query = gr.Textbox(label="Filter tensor names", placeholder="attn, mlp, norm, embd...")
                    editable_only = gr.Checkbox(label="Editable only", value=False)
                    filter_btn = gr.Button("Apply filter")
                tensor_df = gr.Dataframe(label="Tensor list")

            with gr.Tab("Inspect"):
                with gr.Row():
                    inspect_tensor_name = gr.Dropdown(label="Tensor", choices=[])
                    inspect_decode_as = gr.Radio(label="Decode as", choices=["auto", "F32", "F16", "BF16"], value="auto")
                    inspect_max = gr.Number(label="Preview elements", value=32, precision=0)
                    inspect_btn = gr.Button("Inspect", variant="primary")
                inspect_stats = gr.Markdown()
                inspect_preview = gr.Dataframe(label="Flattened preview")

            with gr.Tab("Patch scalar"):
                gr.Markdown(
                    "Patch exactly one number inside the tensor.\n\n"
                    "- **Indices** = coordinates like `0,1,2`\n"
                    "- **New value** = replacement number\n"
                    "- **Add to batch** to queue the operation, or use **Apply now** for immediate save"
                )
                with gr.Row():
                    scalar_tensor_name = gr.Dropdown(label="Tensor", choices=[])
                    scalar_decode_as = gr.Radio(label="Decode as", choices=["auto", "F32", "F16", "BF16"], value="auto")
                with gr.Row():
                    scalar_indices = gr.Textbox(label="Indices (comma-separated)", placeholder="0,1,2")
                    scalar_value = gr.Number(label="New value", value=0.0)
                with gr.Row():
                    scalar_output = gr.Textbox(label="Output GGUF path (for direct apply)")
                    with gr.Column():
                        scalar_add_batch_btn = gr.Button("Add to batch", variant="secondary")
                        scalar_btn = gr.Button("Apply now", variant="primary")
                scalar_add_batch_msg = gr.Markdown()
                scalar_result = gr.Markdown()

            with gr.Tab("Transform whole tensor"):
                gr.Markdown(
                    "Apply one formula to **every value** in the selected tensor.\n\n"
                    "**Math:** `new = old * scale + bias`, then optional clipping.\n\n"
                    "- **Scale** multiplies every value\n"
                    "- **Bias** adds a constant to every value\n"
                    "- **Clip min / max** cap values after scaling + bias\n"
                    "- **Add to batch** to queue, or **Apply now** for direct save"
                )
                with gr.Row():
                    transform_tensor_name = gr.Dropdown(label="Tensor", choices=[])
                    transform_decode_as = gr.Radio(label="Decode as", choices=["auto", "F32", "F16", "BF16"], value="auto")
                with gr.Row():
                    transform_scale = gr.Number(label="Scale", value=1.0)
                    transform_bias = gr.Number(label="Bias", value=0.0)
                    transform_clip_min = gr.Number(label="Clip min (optional)", value=float("nan"))
                    transform_clip_max = gr.Number(label="Clip max (optional)", value=float("nan"))
                transform_output = gr.Textbox(label="Output GGUF path (for direct apply)")
                with gr.Row():
                    preview_transform_btn = gr.Button("Preview transform")
                    transform_add_batch_btn = gr.Button("Add to batch", variant="secondary")
                    transform_btn = gr.Button("Apply now", variant="primary")
                transform_preview_md = gr.Markdown()
                transform_preview_df = gr.Dataframe(label="Before / After preview")
                transform_add_batch_msg = gr.Markdown()
                transform_result = gr.Markdown()

            with gr.Tab("Patch slice"):
                gr.Markdown(
                    "Edit one **slice** of a tensor by selecting an axis and an index.\n\n"
                    "Examples:\n"
                    "- axis `0`, index `5` = the 6th slice along the first dimension\n"
                    "- mode `set_constant` = overwrite the whole slice with one value\n"
                    "- mode `scale_and_bias` = `slice = slice * scale + bias`\n"
                    "- **Add to batch** or **Apply now**"
                )
                with gr.Row():
                    slice_tensor_name = gr.Dropdown(label="Tensor", choices=[])
                    slice_decode_as = gr.Radio(label="Decode as", choices=["auto", "F32", "F16", "BF16"], value="auto")
                with gr.Row():
                    slice_axis = gr.Number(label="Axis", value=0, precision=0)
                    slice_index = gr.Number(label="Index along axis", value=0, precision=0)
                    slice_mode = gr.Radio(label="Mode", choices=["set_constant", "scale_and_bias"], value="set_constant")
                with gr.Row():
                    slice_value = gr.Number(label="Value (used for set_constant)", value=0.0)
                    slice_scale = gr.Number(label="Scale (used for scale_and_bias)", value=1.0)
                    slice_bias = gr.Number(label="Bias (used for scale_and_bias)", value=0.0)
                slice_output = gr.Textbox(label="Output GGUF path (for direct apply)")
                with gr.Row():
                    preview_slice_btn = gr.Button("Preview slice edit")
                    slice_add_batch_btn = gr.Button("Add to batch", variant="secondary")
                    slice_btn = gr.Button("Apply now", variant="primary")
                slice_preview_md = gr.Markdown()
                slice_preview_df = gr.Dataframe(label="Slice before / after preview")
                slice_add_batch_msg = gr.Markdown()
                slice_result = gr.Markdown()

            with gr.Tab("Batch Manager"):
                gr.Markdown(
                    "### Batch Operations\n"
                    "Queue up multiple edits, then apply them all at once to create a single output GGUF.\n\n"
                    "**Workflow:**\n"
                    "1. Use the tabs above to add operations to the batch\n"
                    "2. View the batch queue below\n"
                    "3. Specify the final output path\n"
                    "4. Click **Apply batch** to execute all operations in sequence"
                )
                batch_queue_md = gr.Markdown(value="### Batch Queue\nEmpty. Add operations above to get started.")
                batch_output = gr.Textbox(label="Output GGUF path", placeholder="/path/to/output.batch.gguf")
                with gr.Row():
                    batch_apply_btn = gr.Button("Apply batch", variant="primary", scale=2)
                    batch_clear_btn = gr.Button("Clear batch", variant="stop", scale=1)
                batch_result = gr.Markdown()

        load_btn.click(
            on_load,
            inputs=[gguf_path],
            outputs=[
                manifest_state, summary_md, meta_df, tensor_df,
                inspect_tensor_name, scalar_tensor_name, transform_tensor_name, slice_tensor_name,
                scalar_output, transform_output, slice_output, batch_output,
            ],
        )
        filter_btn.click(filter_tensor_table, inputs=[manifest_state, filter_query, editable_only], outputs=[tensor_df])
        inspect_btn.click(inspect_tensor, inputs=[manifest_state, inspect_tensor_name, inspect_decode_as, inspect_max], outputs=[inspect_stats, inspect_preview])
        
        # Scalar tab
        scalar_btn.click(patch_scalar, inputs=[manifest_state, scalar_tensor_name, scalar_decode_as, scalar_indices, scalar_value, scalar_output], outputs=[scalar_result])
        scalar_add_batch_btn.click(
            batch_add_scalar,
            inputs=[batch_queue, scalar_tensor_name, scalar_decode_as, scalar_indices, scalar_value],
            outputs=[batch_queue, scalar_add_batch_msg],
        )
        
        # Transform tab
        preview_transform_btn.click(preview_transform, inputs=[manifest_state, transform_tensor_name, transform_decode_as, transform_scale, transform_bias, transform_clip_min, transform_clip_max], outputs=[transform_preview_md, transform_preview_df])
        transform_btn.click(patch_transform, inputs=[manifest_state, transform_tensor_name, transform_decode_as, transform_scale, transform_bias, transform_clip_min, transform_clip_max, transform_output], outputs=[transform_result])
        transform_add_batch_btn.click(
            batch_add_transform,
            inputs=[batch_queue, transform_tensor_name, transform_decode_as, transform_scale, transform_bias, transform_clip_min, transform_clip_max],
            outputs=[batch_queue, transform_add_batch_msg],
        )
        
        # Slice tab
        preview_slice_btn.click(preview_slice_edit, inputs=[manifest_state, slice_tensor_name, slice_decode_as, slice_axis, slice_index, slice_mode, slice_value, slice_scale, slice_bias], outputs=[slice_preview_md, slice_preview_df])
        slice_btn.click(patch_slice, inputs=[manifest_state, slice_tensor_name, slice_decode_as, slice_axis, slice_index, slice_mode, slice_value, slice_scale, slice_bias, slice_output], outputs=[slice_result])
        slice_add_batch_btn.click(
            batch_add_slice,
            inputs=[batch_queue, slice_tensor_name, slice_decode_as, slice_axis, slice_index, slice_mode, slice_value, slice_scale, slice_bias],
            outputs=[batch_queue, slice_add_batch_msg],
        )
        
        # Batch manager tab
        batch_apply_btn.click(
            apply_batch,
            inputs=[batch_queue, manifest_state, batch_output],
            outputs=[batch_result],
        )
        batch_clear_btn.click(
            clear_batch,
            inputs=[batch_queue],
            outputs=[batch_queue, batch_queue_md],
        )
        
        # Update batch queue display whenever it changes
        batch_queue.change(
            render_batch_queue,
            inputs=[batch_queue],
            outputs=[batch_queue_md],
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
