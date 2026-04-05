from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np

from .parser import load_gguf
from .tensor_ops import decode_tensor


def compare_gguf(
    original_file,
    patched_file,
    show_unchanged: bool = False,
    elementwise_threshold: int = 1000,
) -> Tuple[str, Any]:
    import pandas as pd

    if original_file is None or patched_file is None:
        raise ValueError("Please upload both original and patched GGUF files.")
    original_path = original_file.name
    patched_path = patched_file.name

    manifest_orig = load_gguf(original_path)
    manifest_patched = load_gguf(patched_path)

    names_orig = {t.name for t in manifest_orig.tensors}
    names_patched = {t.name for t in manifest_patched.tensors}

    all_names = names_orig | names_patched
    only_orig = names_orig - names_patched
    only_patched = names_patched - names_orig
    common = names_orig & names_patched

    results = []
    element_diffs = {}

    for name in sorted(all_names):
        if name in only_orig:
            results.append(
                {
                    "tensor": name,
                    "status": "removed",
                    "shape_orig": str(manifest_orig.tensor_map()[name].shape),
                    "shape_patched": "-",
                    "min_orig": "-",
                    "max_orig": "-",
                    "mean_orig": "-",
                    "std_orig": "-",
                    "min_patched": "-",
                    "max_patched": "-",
                    "mean_patched": "-",
                    "std_patched": "-",
                    "min_delta": "-",
                    "max_delta": "-",
                    "mean_delta": "-",
                }
            )
        elif name in only_patched:
            results.append(
                {
                    "tensor": name,
                    "status": "added",
                    "shape_orig": "-",
                    "shape_patched": str(manifest_patched.tensor_map()[name].shape),
                    "min_orig": "-",
                    "max_orig": "-",
                    "mean_orig": "-",
                    "std_orig": "-",
                    "min_patched": "-",
                    "max_patched": "-",
                    "mean_patched": "-",
                    "std_patched": "-",
                    "min_delta": "-",
                    "max_delta": "-",
                    "mean_delta": "-",
                }
            )
        else:
            tensor_orig = manifest_orig.tensor_map()[name]
            tensor_patched = manifest_patched.tensor_map()[name]

            arr_orig = decode_tensor(original_path, tensor_orig, "auto")
            arr_patched = decode_tensor(patched_path, tensor_patched, "auto")

            flat_orig = arr_orig.reshape(-1).astype(np.float32)
            flat_patched = arr_patched.reshape(-1).astype(np.float32)

            delta = flat_patched - flat_orig
            max_delta = float(np.abs(delta).max())

            if max_delta > 0 or show_unchanged:
                results.append(
                    {
                        "tensor": name,
                        "status": "changed" if max_delta > 0 else "unchanged",
                        "shape_orig": str(tensor_orig.shape),
                        "shape_patched": str(tensor_patched.shape),
                        "min_orig": float(flat_orig.min()),
                        "max_orig": float(flat_orig.max()),
                        "mean_orig": float(flat_orig.mean()),
                        "std_orig": float(flat_orig.std()),
                        "min_patched": float(flat_patched.min()),
                        "max_patched": float(flat_patched.max()),
                        "mean_patched": float(flat_patched.mean()),
                        "std_patched": float(flat_patched.std()),
                        "min_delta": float(delta.min()),
                        "max_delta": float(delta.max()),
                        "mean_delta": float(delta.mean()),
                    }
                )

                n_elements = flat_orig.shape[0]
                if max_delta > 0 and n_elements <= elementwise_threshold:
                    element_diffs[name] = pd.DataFrame(
                        {
                            "index": range(n_elements),
                            "original": flat_orig[:n_elements],
                            "patched": flat_patched[:n_elements],
                            "delta": delta[:n_elements],
                        }
                    )

    if not results:
        return "No differences found.", pd.DataFrame()

    df = pd.DataFrame(results)
    removed_count = sum(1 for r in results if r["status"] == "removed")
    added_count = sum(1 for r in results if r["status"] == "added")
    changed_count = sum(1 for r in results if r["status"] == "changed")

    summary = (
        f"### Comparison Results\n"
        f"- Total tensors: {len(results)}\n"
        f"- Changed: {changed_count}\n"
        f"- Added: {added_count}\n"
        f"- Removed: {removed_count}\n"
    )

    return summary, df
