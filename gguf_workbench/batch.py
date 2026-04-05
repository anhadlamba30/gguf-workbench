from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .parser import GGUFManifest, TensorInfo, manifest_from_dict
from .tensor_ops import (
    decode_tensor,
    encode_tensor,
    parse_indices,
    parse_slice_spec,
    transform_array,
)


@dataclass
class BatchOperation:
    """Represents a single operation to be added to the batch queue."""

    op_type: str
    tensor_name: str
    decode_as: str
    parameters: Dict[str, Any]

    def display_label(self) -> str:
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


def write_tensor_patch(
    manifest: GGUFManifest,
    tensor: TensorInfo,
    arr: np.ndarray,
    decode_as: str,
    output_path: str,
):
    payload = encode_tensor(arr, tensor, decode_as)
    shutil.copyfile(manifest.path, output_path)
    with open(output_path, "r+b") as f:
        f.seek(tensor.abs_offset)
        f.write(payload)


def batch_op_to_dict(op: BatchOperation) -> Dict[str, Any]:
    return {
        "op_type": op.op_type,
        "tensor_name": op.tensor_name,
        "decode_as": op.decode_as,
        "parameters": op.parameters,
    }


def batch_op_from_dict(data: Dict[str, Any]) -> BatchOperation:
    return BatchOperation(
        op_type=data["op_type"],
        tensor_name=data["tensor_name"],
        decode_as=data["decode_as"],
        parameters=data["parameters"],
    )


def batch_add_scalar(
    batch_list: List[Dict[str, Any]],
    tensor_name: str,
    decode_as: str,
    indices_text: str,
    new_value: float,
) -> Tuple[List[Dict[str, Any]], str]:
    if not tensor_name:
        raise ValueError("Please select a tensor")
    op = BatchOperation(
        op_type="scalar",
        tensor_name=tensor_name,
        decode_as=decode_as,
        parameters={"indices": indices_text, "new_value": new_value},
    )
    batch_list.append(batch_op_to_dict(op))
    msg = f"✓ Added: {op.display_label()}"
    return batch_list, msg


def batch_add_transform(
    batch_list: List[Dict[str, Any]],
    tensor_name: str,
    decode_as: str,
    scale: float,
    bias: float,
    clip_min: Optional[float],
    clip_max: Optional[float],
) -> Tuple[List[Dict[str, Any]], str]:
    if not tensor_name:
        raise ValueError("Please select a tensor")
    op = BatchOperation(
        op_type="transform",
        tensor_name=tensor_name,
        decode_as=decode_as,
        parameters={
            "scale": scale,
            "bias": bias,
            "clip_min": clip_min,
            "clip_max": clip_max,
        },
    )
    batch_list.append(batch_op_to_dict(op))
    msg = f"✓ Added: {op.display_label()}"
    return batch_list, msg


def batch_add_slice(
    batch_list: List[Dict[str, Any]],
    tensor_name: str,
    decode_as: str,
    axis: int,
    index: int,
    mode: str,
    value: float,
    scale: float,
    bias: float,
) -> Tuple[List[Dict[str, Any]], str]:
    if not tensor_name:
        raise ValueError("Please select a tensor")
    op = BatchOperation(
        op_type="slice",
        tensor_name=tensor_name,
        decode_as=decode_as,
        parameters={
            "axis": axis,
            "index": index,
            "mode": mode,
            "value": value,
            "scale": scale,
            "bias": bias,
        },
    )
    batch_list.append(batch_op_to_dict(op))
    msg = f"✓ Added: {op.display_label()}"
    return batch_list, msg


def render_batch_queue(batch_list: List[Dict[str, Any]]) -> str:
    if not batch_list:
        return "### Batch Queue\nEmpty. Add operations above to get started."

    lines = ["### Batch Queue", f"**{len(batch_list)} operation(s) pending:**\n"]
    for i, op_dict in enumerate(batch_list, 1):
        op = batch_op_from_dict(op_dict)
        lines.append(f"{i}. {op.display_label()}")
    return "\n".join(lines)


def apply_batch(
    batch_list: List[Dict[str, Any]],
    manifest_dict: Dict[str, Any],
    output_path: str,
    overwrite_confirm: bool = False,
    progress=None,
) -> str:
    if not batch_list:
        raise ValueError("Batch is empty. Add operations first.")
    if not output_path:
        raise ValueError("Please specify an output path.")

    manifest = manifest_from_dict(manifest_dict)
    from .validation import validate_output_path

    validated_path, warning = validate_output_path(
        manifest.path, output_path, overwrite_confirm
    )

    if progress is not None:
        progress(0, desc="Copying original file...")
    shutil.copyfile(manifest.path, validated_path)

    total = len(batch_list)

    for i, op_dict in enumerate(batch_list, 1):
        op = batch_op_from_dict(op_dict)
        tensor = manifest.tensor_map().get(op.tensor_name)
        if tensor is None:
            raise ValueError(f"Operation {i}: Tensor '{op.tensor_name}' not found")

        if progress is not None:
            progress(i / total, desc=f"{i}/{total}: {op.display_label()}")

        arr = decode_tensor(validated_path, tensor, op.decode_as)

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
                slice_view, slicer = parse_slice_spec(
                    arr, int(op.parameters["axis"]), int(op.parameters["index"])
                )
                if op.parameters["mode"] == "set_constant":
                    arr[slicer] = np.float32(op.parameters["value"])
                else:
                    arr[slicer] = slice_view * np.float32(
                        op.parameters["scale"]
                    ) + np.float32(op.parameters["bias"])
            else:
                raise ValueError(f"Operation {i}: Unknown operation type {op.op_type}")

            write_tensor_patch(manifest, tensor, arr, op.decode_as, validated_path)
        except Exception as e:
            raise ValueError(
                f"Operation {i} ({op.op_type} on {op.tensor_name}) failed: {str(e)}"
            )

    if progress is not None:
        progress(1.0, desc="Done!")

    return (
        warning
        + f"✓ **Batch applied successfully!**\n\nApplied {len(batch_list)} operation(s).\n\nSaved to: `{validated_path}`"
    )


def clear_batch(batch_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    return [], "### Batch Queue\nEmpty. Add operations above to get started."
