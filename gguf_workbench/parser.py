from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .constants import (
    GGUF_MAGIC,
    SUPPORTED_VERSIONS,
    GGUF_TYPE_UINT8,
    GGUF_TYPE_INT8,
    GGUF_TYPE_UINT16,
    GGUF_TYPE_INT16,
    GGUF_TYPE_UINT32,
    GGUF_TYPE_INT32,
    GGUF_TYPE_FLOAT32,
    GGUF_TYPE_BOOL,
    GGUF_TYPE_STRING,
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_UINT64,
    GGUF_TYPE_INT64,
    GGUF_TYPE_FLOAT64,
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_NAMES,
    DEFAULT_ALIGNMENT,
)


def align_offset(offset: int, alignment: int) -> int:
    return (offset + alignment - 1) // alignment * alignment


class GGUFParseError(RuntimeError):
    pass


class BinaryReader:
    def __init__(self, data: bytes | mmap.mmap):
        self.data = data
        self.pos = 0
        self._size = len(data)

    def tell(self) -> int:
        return self.pos

    def read(self, n: int) -> bytes:
        end = self.pos + n
        if end > self._size:
            raise GGUFParseError(
                f"Unexpected EOF while reading {n} bytes at offset {self.pos}"
            )
        out = bytes(self.data[self.pos : end])
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
        from .constants import EDITABLE_TYPE_NAMES

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
        from .constants import EDITABLE_TYPE_NAMES

        return [t.name for t in self.tensors if t.editable_kind in EDITABLE_TYPE_NAMES]


class GGUFParser:
    def __init__(self, file_path: str | Path):
        self.path = Path(file_path)
        self._fd = open(self.path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.reader = BinaryReader(self._mm)

    def parse(self) -> GGUFManifest:
        r = self.reader
        magic = r.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFParseError(f"Not a GGUF file: magic={magic!r}")
        version = r.u32()
        if version not in SUPPORTED_VERSIONS:
            raise GGUFParseError(
                f"Unsupported GGUF version {version}. Supported: {sorted(SUPPORTED_VERSIONS)}"
            )
        n_tensors = r.u64()
        n_kv = r.u64()

        metadata: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            value_type = r.u32()
            metadata[key] = self._read_value(r, value_type)

        alignment = int(
            metadata.get("general.alignment", DEFAULT_ALIGNMENT) or DEFAULT_ALIGNMENT
        )
        tensor_descs: List[Dict[str, Any]] = []
        for _ in range(n_tensors):
            name = r.string()
            n_dimensions = r.u32()
            dims_stored = tuple(r.u64() for _ in range(n_dimensions))
            ggml_type = r.u32()
            offset = r.u64()
            tensor_descs.append(
                {
                    "name": name,
                    "dims_stored": dims_stored,
                    "shape": tuple(reversed(dims_stored)),
                    "ggml_type": ggml_type,
                    "offset": offset,
                }
            )

        tensor_data_start = align_offset(r.tell(), alignment)
        file_size = self._mm.size()
        tensors: List[TensorInfo] = []
        for desc in tensor_descs:
            abs_off = tensor_data_start + int(desc["offset"])
            n_elements = (
                int(np.prod(desc["shape"], dtype=np.int64)) if desc["shape"] else 1
            )
            tensors.append(
                TensorInfo(
                    name=desc["name"],
                    shape=tuple(int(x) for x in desc["shape"]),
                    stored_dims=tuple(int(x) for x in desc["dims_stored"]),
                    ggml_type=int(desc["ggml_type"]),
                    ggml_type_name=GGML_TYPE_NAMES.get(
                        int(desc["ggml_type"]), f"TYPE_{desc['ggml_type']}"
                    ),
                    offset=int(desc["offset"]),
                    abs_offset=int(abs_off),
                    n_elements=n_elements,
                    n_bytes=0,
                    editable_kind="UNKNOWN",
                )
            )

        tensors_sorted = sorted(tensors, key=lambda t: t.abs_offset)
        for i, tensor in enumerate(tensors_sorted):
            next_abs = (
                tensors_sorted[i + 1].abs_offset
                if i + 1 < len(tensors_sorted)
                else file_size
            )
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
        if (
            tensor.ggml_type == GGML_TYPE_F32
            and tensor.n_bytes == tensor.n_elements * 4
        ):
            return "F32"
        if (
            tensor.ggml_type == GGML_TYPE_F16
            and tensor.n_bytes == tensor.n_elements * 2
        ):
            return "F16"
        if (
            tensor.ggml_type == GGML_TYPE_BF16
            and tensor.n_bytes == tensor.n_elements * 2
        ):
            return "BF16"
        if tensor.n_bytes == tensor.n_elements * 4:
            return "F32"
        if tensor.n_bytes == tensor.n_elements * 2:
            return "F16_OR_BF16"
        return GGML_TYPE_NAMES.get(tensor.ggml_type, f"TYPE_{tensor.ggml_type}")

    def close(self) -> None:
        if hasattr(self, "_mm") and not self._mm.closed:
            self._mm.close()
        if hasattr(self, "_fd") and not self._fd.closed:
            self._fd.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def load_gguf(file_path: str) -> GGUFManifest:
    with GGUFParser(file_path) as parser:
        return parser.parse()


def load_manifest(file_path: str) -> Tuple[GGUFManifest, Any, Any, List[str]]:
    import pandas as pd
    from pathlib import Path

    with GGUFParser(file_path) as parser:
        manifest = parser.parse()
    meta_rows = []
    for k, v in manifest.metadata.items():
        preview = v
        if isinstance(v, list) and len(v) > 16:
            preview = v[:16] + [f"... ({len(v)} total)"]
        meta_rows.append({"key": k, "python_type": type(v).__name__, "value": preview})
    tensor_rows = [t.to_row() for t in manifest.tensors]
    meta_df = pd.DataFrame(meta_rows)
    tensor_df = pd.DataFrame(tensor_rows)
    choices = manifest.editable_tensors()
    return manifest, meta_df, tensor_df, choices


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
        raise ValueError("No GGUF loaded yet.")
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
    from .constants import EDITABLE_TYPE_NAMES

    editable = sum(
        1 for t in manifest.tensors if t.editable_kind in EDITABLE_TYPE_NAMES
    )
    return (
        f"### {Path(manifest.path).name}\n"
        f"- Version: **{manifest.version}**\n"
        f"- File size: **{manifest.file_size / (1024**3):.4f} GiB**\n"
        f"- Metadata entries: **{manifest.n_kv}**\n"
        f"- Tensors: **{manifest.n_tensors}**\n"
        f"- Alignment: **{manifest.alignment}**\n"
        f"- Editable tensors (float-like): **{editable}**"
    )
