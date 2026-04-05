from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .constants import APP_TITLE
from .parser import (
    load_manifest,
    manifest_from_dict,
    manifest_summary,
    manifest_to_dict,
)
from .tensor_ops import (
    build_transform_preview,
    decode_tensor,
    parse_indices,
    parse_slice_spec,
    transform_array,
)
from .validation import default_output_path, validate_output_path
from .compare import compare_gguf
from .batch import (
    apply_batch,
    batch_add_scalar,
    batch_add_slice,
    batch_add_transform,
    clear_batch,
    render_batch_queue,
)
from .mri_viz import (
    compute_tensor_stats,
    plot_histogram,
    plot_heatmap,
    plot_layer_summary,
    plot_model_overview,
    get_tensor_quantization_info,
)


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


def inspect_tensor(
    manifest_dict: Dict[str, Any], tensor_name: str, decode_as: str, max_items: int
):
    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map().get(tensor_name)
    if tensor is None:
        raise ValueError(f"Tensor not found: {tensor_name}")
    arr = decode_tensor(manifest.path, tensor, decode_as)
    flat = arr.reshape(-1)
    n = min(int(max_items), flat.shape[0])
    preview = pd.DataFrame(
        [{"flat_index": i, "value": float(v)} for i, v in enumerate(flat[:n])]
    )
    from .tensor_ops import resolve_decode_kind

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


def inspect_tensor_mri(manifest_dict: Dict[str, Any], tensor_name: str, decode_as: str):
    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map().get(tensor_name)
    if tensor is None:
        raise ValueError(f"Tensor not found: {tensor_name}")
    arr = decode_tensor(manifest.path, tensor, decode_as)

    quant_info = get_tensor_quantization_info(tensor)
    stats = compute_tensor_stats(arr)

    hist_fig = plot_histogram(
        arr,
        title=f"{tensor.name} - Weight Distribution ({quant_info})",
        n_bins=80,
        show_kde=True,
    )

    heatmap_fig = None
    if arr.ndim >= 2:
        heatmap_fig = plot_heatmap(
            arr,
            title=f"{tensor.name} - Weight Matrix Heatmap",
        )
    else:
        heatmap_fig = go.Figure().update_layout(
            title="Heatmap not available for 1D tensors"
        )

    info_text = (
        f"### {tensor.name}\n"
        f"**Type:** {quant_info}\n"
        f"**Shape:** {stats['shape']}\n"
        f"**Elements:** {stats['n_elements']:,}\n\n"
        f"**Statistics:**\n"
        f"- Min: {stats['min']:.6g}\n"
        f"- Max: {stats['max']:.6g}\n"
        f"- Mean: {stats['mean']:.6g}\n"
        f"- Std: {stats['std']:.6g}"
    )

    return hist_fig, heatmap_fig, info_text


def mri_layer_summary(manifest_dict: Dict[str, Any]):
    fig, _ = plot_layer_summary(manifest_dict)
    return fig


def mri_model_overview(manifest_dict: Dict[str, Any]):
    fig = plot_model_overview(manifest_dict)
    return fig


def mri_get_all_tensor_choices(manifest_dict: Dict[str, Any]) -> List[str]:
    manifest = manifest_from_dict(manifest_dict)
    return [t.name for t in manifest.tensors]


def on_load(gguf_path: str = None, gguf_file=None):
    file_path = None
    if gguf_path and gguf_path.strip():
        candidate = gguf_path.strip()
        if Path(candidate).exists():
            file_path = candidate
    if file_path is None and gguf_file is not None:
        if isinstance(gguf_file, str):
            file_path = gguf_file
        else:
            file_path = gguf_file.name
    if not file_path:
        raise ValueError("Please provide a valid GGUF file path or upload a file.")
    manifest, meta_df, tensor_df, choices = load_manifest(file_path)
    manifest_dict = manifest_to_dict(manifest)
    summary = manifest_summary(manifest)
    all_tensor_choices = [t.name for t in manifest.tensors]
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
        gr.update(choices=all_tensor_choices, value=default_tensor),
        gr.update(choices=all_tensor_choices, value=default_tensor),
        gr.update(choices=all_tensor_choices, value=default_tensor),
        gr.update(choices=all_tensor_choices, value=default_tensor),
        gr.update(choices=all_tensor_choices, value=default_tensor),
        scalar_out,
        transform_out,
        slice_out,
        batch_out,
    )


def patch_scalar(
    manifest_dict: Dict[str, Any],
    tensor_name: str,
    decode_as: str,
    indices_text: str,
    new_value: float,
    output_path: str,
    overwrite_confirm: bool = False,
):
    from .batch import write_tensor_patch

    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map()[tensor_name]
    output_path = output_path or default_output_path(manifest.path)
    validated_path, warning = validate_output_path(
        manifest.path, output_path, overwrite_confirm
    )
    arr = decode_tensor(manifest.path, tensor, decode_as)
    idxs = parse_indices(indices_text, arr.shape)
    before = float(arr[idxs])
    arr[idxs] = np.float32(new_value)
    write_tensor_patch(manifest, tensor, arr, decode_as, validated_path)
    return (
        warning
        + f"Patched `{tensor.name}{idxs}` from **{before:.8g}** to **{float(new_value):.8g}**\n\nSaved: `{validated_path}`"
    )


def preview_transform(
    manifest_dict: Dict[str, Any],
    tensor_name: str,
    decode_as: str,
    scale: float,
    bias: float,
    clip_min: Optional[float],
    clip_max: Optional[float],
):
    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map()[tensor_name]
    before = decode_tensor(manifest.path, tensor, decode_as)
    after = transform_array(before, scale, bias, clip_min, clip_max)
    return build_transform_preview(before, after)


def patch_transform(
    manifest_dict: Dict[str, Any],
    tensor_name: str,
    decode_as: str,
    scale: float,
    bias: float,
    clip_min: Optional[float],
    clip_max: Optional[float],
    output_path: str,
    overwrite_confirm: bool = False,
):
    from .batch import write_tensor_patch

    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map()[tensor_name]
    output_path = output_path or default_output_path(manifest.path, ".transformed.gguf")
    validated_path, warning = validate_output_path(
        manifest.path, output_path, overwrite_confirm
    )
    before = decode_tensor(manifest.path, tensor, decode_as)
    after = transform_array(before, scale, bias, clip_min, clip_max)
    write_tensor_patch(manifest, tensor, after, decode_as, validated_path)
    preview_text, _ = build_transform_preview(before, after)
    return warning + preview_text + f"\n\nSaved: `{validated_path}`"


def preview_slice_edit(
    manifest_dict: Dict[str, Any],
    tensor_name: str,
    decode_as: str,
    axis: int,
    index: int,
    mode: str,
    value: float,
    scale: float,
    bias: float,
):
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


def patch_slice(
    manifest_dict: Dict[str, Any],
    tensor_name: str,
    decode_as: str,
    axis: int,
    index: int,
    mode: str,
    value: float,
    scale: float,
    bias: float,
    output_path: str,
    overwrite_confirm: bool = False,
):
    from .batch import write_tensor_patch

    manifest = manifest_from_dict(manifest_dict)
    tensor = manifest.tensor_map()[tensor_name]
    output_path = output_path or default_output_path(manifest.path, ".slice.gguf")
    validated_path, warning = validate_output_path(
        manifest.path, output_path, overwrite_confirm
    )
    arr = decode_tensor(manifest.path, tensor, decode_as)
    slice_view, slicer = parse_slice_spec(arr, int(axis), int(index))
    before = np.array(slice_view, copy=True)
    if mode == "set_constant":
        arr[slicer] = np.float32(value)
    else:
        arr[slicer] = slice_view * np.float32(scale) + np.float32(bias)
    after = np.array(arr[slicer], copy=True)
    write_tensor_patch(manifest, tensor, arr, decode_as, validated_path)
    preview_text, _ = build_transform_preview(before, after)
    return (
        warning
        + f"### Slice Patch Applied (axis={axis}, index={index})\n\n"
        + preview_text
        + f"\n\nSaved: `{validated_path}`"
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
            with gr.Column(scale=2):
                gguf_path = gr.Textbox(
                    label="GGUF file path",
                    placeholder="/path/to/model.gguf",
                )
            with gr.Column(scale=1):
                gguf_file = gr.File(
                    label="Or upload file",
                    file_count="single",
                    file_types=[".gguf"],
                )
            load_btn = gr.Button("Load GGUF", variant="primary", scale=1)

        summary_md = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("Metadata + Tensors"):
                meta_df = gr.Dataframe(label="Metadata")
                with gr.Row():
                    filter_query = gr.Textbox(
                        label="Filter tensor names",
                        placeholder="attn, mlp, norm, embd...",
                    )
                    editable_only = gr.Checkbox(label="Editable only", value=False)
                    filter_btn = gr.Button("Apply filter")
                tensor_df = gr.Dataframe(label="Tensor list")

            with gr.Tab("Inspect"):
                with gr.Row():
                    inspect_tensor_name = gr.Dropdown(label="Tensor", choices=[])
                    inspect_decode_as = gr.Radio(
                        label="Decode as",
                        choices=["auto", "F32", "F16", "BF16"],
                        value="auto",
                    )
                    inspect_max = gr.Number(
                        label="Preview elements", value=32, precision=0
                    )
                    inspect_mri_mode = gr.Checkbox(
                        label="MRI Mode (Visualization)", value=False
                    )
                    inspect_btn = gr.Button("Inspect", variant="primary")
                with gr.Row(visible=False) as inspect_basic_view:
                    inspect_stats = gr.Markdown()
                    inspect_preview = gr.Dataframe(label="Flattened preview")
                with gr.Row(visible=False) as inspect_mri_view:
                    with gr.Column(scale=1):
                        inspect_mri_info = gr.Markdown()
                    with gr.Column(scale=2):
                        inspect_mri_hist = gr.Plot()
                        inspect_mri_heatmap = gr.Plot()

            with gr.Tab("MRI Mode"):
                gr.Markdown(
                    "### 🧠 MRI Mode - Neural Weight Explorer\n"
                    "Interactive visualizations of tensor weight distributions and patterns.\n\n"
                    "**Tip:** Click on any tensor in the layer selector to explore its weights."
                )
                with gr.Row():
                    mri_tensor_select = gr.Dropdown(
                        label="Select Tensor",
                        choices=[],
                        scale=3,
                    )
                    mri_decode_as = gr.Radio(
                        label="Decode as",
                        choices=["auto", "F32", "F16", "BF16"],
                        value="auto",
                        scale=1,
                    )
                    mri_load_btn = gr.Button("Load MRI", variant="primary", scale=1)
                with gr.Row():
                    mri_layer_summary_plot = gr.Plot(label="Layer Summary")
                with gr.Row():
                    mri_model_overview_plot = gr.Plot(label="Model Overview")
                with gr.Row():
                    mri_hist_plot = gr.Plot(label="Weight Distribution")
                    mri_heatmap_plot = gr.Plot(label="Weight Matrix Heatmap")
                mri_tensor_info = gr.Markdown()

            with gr.Tab("Patch scalar"):
                gr.Markdown(
                    "Patch exactly one number inside the tensor.\n\n"
                    "- **Indices** = coordinates like `0,1,2`\n"
                    "- **New value** = replacement number\n"
                    "- **Add to batch** to queue the operation, or use **Apply now** for immediate save"
                )
                with gr.Row():
                    scalar_tensor_name = gr.Dropdown(label="Tensor", choices=[])
                    scalar_decode_as = gr.Radio(
                        label="Decode as",
                        choices=["auto", "F32", "F16", "BF16"],
                        value="auto",
                    )
                with gr.Row():
                    scalar_indices = gr.Textbox(
                        label="Indices (comma-separated)", placeholder="0,1,2"
                    )
                    scalar_value = gr.Number(label="New value", value=0.0)
                with gr.Row():
                    scalar_output = gr.Textbox(
                        label="Output GGUF path (for direct apply)"
                    )
                    with gr.Column():
                        scalar_add_batch_btn = gr.Button(
                            "Add to batch", variant="secondary"
                        )
                        scalar_overwrite_confirm = gr.Checkbox(
                            label="Confirm overwrite", value=False
                        )
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
                    transform_decode_as = gr.Radio(
                        label="Decode as",
                        choices=["auto", "F32", "F16", "BF16"],
                        value="auto",
                    )
                with gr.Row():
                    transform_scale = gr.Number(label="Scale", value=1.0)
                    transform_bias = gr.Number(label="Bias", value=0.0)
                    transform_clip_min = gr.Number(
                        label="Clip min (optional)", value=float("nan")
                    )
                    transform_clip_max = gr.Number(
                        label="Clip max (optional)", value=float("nan")
                    )
                transform_output = gr.Textbox(
                    label="Output GGUF path (for direct apply)"
                )
                with gr.Row():
                    preview_transform_btn = gr.Button("Preview transform")
                    transform_add_batch_btn = gr.Button(
                        "Add to batch", variant="secondary"
                    )
                    transform_overwrite_confirm = gr.Checkbox(
                        label="Confirm overwrite", value=False
                    )
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
                    slice_decode_as = gr.Radio(
                        label="Decode as",
                        choices=["auto", "F32", "F16", "BF16"],
                        value="auto",
                    )
                with gr.Row():
                    slice_axis = gr.Number(label="Axis", value=0, precision=0)
                    slice_index = gr.Number(
                        label="Index along axis", value=0, precision=0
                    )
                    slice_mode = gr.Radio(
                        label="Mode",
                        choices=["set_constant", "scale_and_bias"],
                        value="set_constant",
                    )
                with gr.Row():
                    slice_value = gr.Number(
                        label="Value (used for set_constant)", value=0.0
                    )
                    slice_scale = gr.Number(
                        label="Scale (used for scale_and_bias)", value=1.0
                    )
                    slice_bias = gr.Number(
                        label="Bias (used for scale_and_bias)", value=0.0
                    )
                slice_output = gr.Textbox(label="Output GGUF path (for direct apply)")
                with gr.Row():
                    preview_slice_btn = gr.Button("Preview slice edit")
                    slice_add_batch_btn = gr.Button("Add to batch", variant="secondary")
                    slice_overwrite_confirm = gr.Checkbox(
                        label="Confirm overwrite", value=False
                    )
                    slice_btn = gr.Button("Apply now", variant="primary")
                slice_preview_md = gr.Markdown()
                slice_preview_df = gr.Dataframe(label="Slice before / after preview")
                slice_add_batch_msg = gr.Markdown()
                slice_result = gr.Markdown()

            with gr.Tab("Compare"):
                gr.Markdown(
                    "### Diff View\n"
                    "Compare two GGUF files to see what changed.\n\n"
                    "Load an original file and a patched file to see tensor-level differences."
                )
                with gr.Row():
                    compare_original = gr.File(
                        label="Original GGUF",
                        file_count="single",
                        file_types=[".gguf"],
                    )
                    compare_patched = gr.File(
                        label="Patched GGUF",
                        file_count="single",
                        file_types=[".gguf"],
                    )
                    compare_btn = gr.Button("Compare", variant="primary")
                with gr.Row():
                    compare_show_unchanged = gr.Checkbox(
                        label="Show unchanged tensors", value=False
                    )
                    compare_threshold = gr.Number(
                        label="Element-wise diff threshold",
                        value=1000,
                        precision=0,
                        info="Max elements to show element-wise diff",
                    )
                compare_result = gr.Markdown()
                compare_diff_df = gr.Dataframe(label="Tensor diffs")

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
                batch_queue_md = gr.Markdown(
                    value="### Batch Queue\nEmpty. Add operations above to get started."
                )
                batch_output = gr.Textbox(
                    label="Output GGUF path", placeholder="/path/to/output.batch.gguf"
                )
                batch_overwrite_confirm = gr.Checkbox(
                    label="Confirm overwrite", value=False
                )
                with gr.Row():
                    batch_apply_btn = gr.Button(
                        "Apply batch", variant="primary", scale=2
                    )
                    batch_clear_btn = gr.Button("Clear batch", variant="stop", scale=1)
                batch_result = gr.Markdown()

        load_btn.click(
            on_load,
            inputs=[gguf_path, gguf_file],
            outputs=[
                manifest_state,
                summary_md,
                meta_df,
                tensor_df,
                inspect_tensor_name,
                scalar_tensor_name,
                transform_tensor_name,
                slice_tensor_name,
                mri_tensor_select,
                scalar_output,
                transform_output,
                slice_output,
                batch_output,
            ],
        )
        filter_btn.click(
            filter_tensor_table,
            inputs=[manifest_state, filter_query, editable_only],
            outputs=[tensor_df],
        )

        def update_mri_visibility(enabled):
            if enabled:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
            return (
                gr.update(visible=True),
                gr.update(visible=False),
            )

        inspect_mri_mode.change(
            update_mri_visibility,
            inputs=[inspect_mri_mode],
            outputs=[inspect_basic_view, inspect_mri_view],
        )

        def handle_inspect_click(
            mri_enabled, manifest_dict, tensor_name, decode_as, max_items
        ):
            if mri_enabled:
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                )
            return (
                gr.update(visible=True),
                gr.update(visible=False),
            )

        inspect_btn.click(
            handle_inspect_click,
            inputs=[
                inspect_mri_mode,
                manifest_state,
                inspect_tensor_name,
                inspect_decode_as,
                inspect_max,
            ],
            outputs=[inspect_basic_view, inspect_mri_view],
        )

        inspect_btn.click(
            inspect_tensor,
            inputs=[
                manifest_state,
                inspect_tensor_name,
                inspect_decode_as,
                inspect_max,
            ],
            outputs=[inspect_stats, inspect_preview],
        )

        inspect_btn.click(
            inspect_tensor_mri,
            inputs=[
                manifest_state,
                inspect_tensor_name,
                inspect_decode_as,
            ],
            outputs=[inspect_mri_hist, inspect_mri_heatmap, inspect_mri_info],
            show_progress=True,
        )

        scalar_btn.click(
            patch_scalar,
            inputs=[
                manifest_state,
                scalar_tensor_name,
                scalar_decode_as,
                scalar_indices,
                scalar_value,
                scalar_output,
                scalar_overwrite_confirm,
            ],
            outputs=[scalar_result],
        )
        scalar_add_batch_btn.click(
            batch_add_scalar,
            inputs=[
                batch_queue,
                scalar_tensor_name,
                scalar_decode_as,
                scalar_indices,
                scalar_value,
            ],
            outputs=[batch_queue, scalar_add_batch_msg],
        )

        preview_transform_btn.click(
            preview_transform,
            inputs=[
                manifest_state,
                transform_tensor_name,
                transform_decode_as,
                transform_scale,
                transform_bias,
                transform_clip_min,
                transform_clip_max,
            ],
            outputs=[transform_preview_md, transform_preview_df],
        )
        transform_btn.click(
            patch_transform,
            inputs=[
                manifest_state,
                transform_tensor_name,
                transform_decode_as,
                transform_scale,
                transform_bias,
                transform_clip_min,
                transform_clip_max,
                transform_output,
                transform_overwrite_confirm,
            ],
            outputs=[transform_result],
        )
        transform_add_batch_btn.click(
            batch_add_transform,
            inputs=[
                batch_queue,
                transform_tensor_name,
                transform_decode_as,
                transform_scale,
                transform_bias,
                transform_clip_min,
                transform_clip_max,
            ],
            outputs=[batch_queue, transform_add_batch_msg],
        )

        preview_slice_btn.click(
            preview_slice_edit,
            inputs=[
                manifest_state,
                slice_tensor_name,
                slice_decode_as,
                slice_axis,
                slice_index,
                slice_mode,
                slice_value,
                slice_scale,
                slice_bias,
            ],
            outputs=[slice_preview_md, slice_preview_df],
        )
        slice_btn.click(
            patch_slice,
            inputs=[
                manifest_state,
                slice_tensor_name,
                slice_decode_as,
                slice_axis,
                slice_index,
                slice_mode,
                slice_value,
                slice_scale,
                slice_bias,
                slice_output,
                slice_overwrite_confirm,
            ],
            outputs=[slice_result],
        )
        slice_add_batch_btn.click(
            batch_add_slice,
            inputs=[
                batch_queue,
                slice_tensor_name,
                slice_decode_as,
                slice_axis,
                slice_index,
                slice_mode,
                slice_value,
                slice_scale,
                slice_bias,
            ],
            outputs=[batch_queue, slice_add_batch_msg],
        )

        batch_apply_btn.click(
            apply_batch,
            inputs=[batch_queue, manifest_state, batch_output, batch_overwrite_confirm],
            outputs=[batch_result],
            show_progress=True,
        )
        batch_clear_btn.click(
            clear_batch,
            inputs=[batch_queue],
            outputs=[batch_queue, batch_queue_md],
        )

        compare_btn.click(
            compare_gguf,
            inputs=[
                compare_original,
                compare_patched,
                compare_show_unchanged,
                compare_threshold,
            ],
            outputs=[compare_result, compare_diff_df],
            show_progress=True,
        )

        load_btn.click(
            mri_get_all_tensor_choices,
            inputs=[manifest_state],
            outputs=[mri_tensor_select],
        )

        load_btn.click(
            mri_layer_summary,
            inputs=[manifest_state],
            outputs=[mri_layer_summary_plot],
        )

        load_btn.click(
            mri_model_overview,
            inputs=[manifest_state],
            outputs=[mri_model_overview_plot],
        )

        mri_load_btn.click(
            inspect_tensor_mri,
            inputs=[
                manifest_state,
                mri_tensor_select,
                mri_decode_as,
            ],
            outputs=[mri_hist_plot, mri_heatmap_plot, mri_tensor_info],
            show_progress=True,
        )

        batch_queue.change(
            render_batch_queue,
            inputs=[batch_queue],
            outputs=[batch_queue_md],
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
