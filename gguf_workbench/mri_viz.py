from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_tensor_stats(arr: np.ndarray) -> Dict[str, Any]:
    flat = arr.reshape(-1).astype(np.float64)
    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "n_elements": flat.shape[0],
        "shape": list(arr.shape),
    }


def plot_histogram(
    arr: np.ndarray,
    title: str = "Weight Distribution",
    n_bins: int = 100,
    show_kde: bool = True,
    show_stats: bool = True,
    subsample: Optional[int] = None,
) -> go.Figure:
    flat = arr.reshape(-1).astype(np.float64)
    if subsample is not None and flat.shape[0] > subsample:
        indices = np.random.choice(flat.shape[0], subsample, replace=False)
        flat = flat[indices]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=flat,
            nbinsx=n_bins,
            name="Histogram",
            marker_color="rgba(100, 149, 237, 0.7)",
            opacity=0.8,
        )
    )

    if show_kde:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(flat, bw_method="scott")
            x_range = np.linspace(float(flat.min()), float(flat.max()), 500)
            y_kde = kde(x_range) * len(flat) * (flat.max() - flat.min()) / n_bins
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde,
                    mode="lines",
                    name="KDE",
                    line=dict(color="rgba(255, 99, 71, 0.9)", width=2),
                )
            )
        except Exception:
            pass

    mean_val = float(flat.mean())
    std_val = float(flat.std())
    min_val = float(flat.min())
    max_val = float(flat.max())

    fig.add_vline(
        x=mean_val, line_dash="dash", line_color="green", annotation_text="mean"
    )
    fig.add_vline(
        x=mean_val + std_val,
        line_dash="dot",
        line_color="orange",
        annotation_text="+1σ",
    )
    fig.add_vline(
        x=mean_val - std_val,
        line_dash="dot",
        line_color="orange",
        annotation_text="-1σ",
    )
    fig.add_vline(
        x=mean_val + 2 * std_val,
        line_dash="dot",
        line_color="red",
        annotation_text="+2σ",
    )
    fig.add_vline(
        x=mean_val - 2 * std_val,
        line_dash="dot",
        line_color="red",
        annotation_text="-2σ",
    )

    if show_stats:
        stats_text = (
            f"Min: {min_val:.4f}<br>"
            f"Max: {max_val:.4f}<br>"
            f"Mean: {mean_val:.4f}<br>"
            f"Std: {std_val:.4f}"
        )
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        )

    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        template="plotly_white",
        height=400,
        hovermode="x unified",
    )
    return fig


def plot_heatmap(
    arr: np.ndarray,
    title: str = "Weight Matrix Heatmap",
    normalize: Optional[str] = None,
    max_size: int = 512,
) -> go.Figure:
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.shape[0] > max_size or arr.shape[1] > max_size:
        step_r = max(1, arr.shape[0] // max_size)
        step_c = max(1, arr.shape[1] // max_size)
        arr = arr[::step_r, ::step_c]

    display_arr = arr.copy()
    if normalize == "row":
        row_means = display_arr.mean(axis=1, keepdims=True)
        display_arr = display_arr - row_means
    elif normalize == "column":
        col_means = display_arr.mean(axis=0, keepdims=True)
        display_arr = display_arr - col_means
    elif normalize == "abs":
        display_arr = np.abs(display_arr)

    vmin = float(display_arr.min())
    vmax = float(display_arr.max())

    fig = go.Figure(
        data=go.Heatmap(
            z=display_arr,
            colorscale="Viridis",
            zmin=vmin,
            zmax=vmax,
            hovertemplate="row: %{y}<br>col: %{x}<br>value: %{z:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Row",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_layer_summary(
    manifest_dict: Dict[str, Any],
) -> Tuple[go.Figure, List[Dict[str, Any]]]:
    from .parser import manifest_from_dict

    manifest = manifest_from_dict(manifest_dict)
    layer_data = []

    for tensor in manifest.tensors:
        shape_str = "×".join(map(str, tensor.shape))
        layer_data.append(
            {
                "name": tensor.name,
                "shape": shape_str,
                "type": tensor.ggml_type_name,
                "elements": tensor.n_elements,
                "bytes": tensor.n_bytes,
                "editable": tensor.editable_kind in {"F32", "F16", "BF16"},
            }
        )

    df = layer_data
    types = list(set(d["type"] for d in df))
    type_counts = {t: sum(1 for d in df if d["type"] == t) for t in types}

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Tensor Count by Type", "Tensor Size Distribution"),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
    )

    fig.add_trace(
        go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            hole=0.4,
        ),
        row=1,
        col=1,
    )

    names = [
        d["name"][:40] + "..." if len(d["name"]) > 40 else d["name"] for d in df[:30]
    ]
    sizes = [d["bytes"] / (1024 * 1024) for d in df[:30]]
    fig.add_trace(
        go.Bar(
            x=names,
            y=sizes,
            marker_color="steelblue",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Model Tensor Overview",
        template="plotly_white",
        height=400,
        showlegend=False,
    )

    return fig, layer_data


def plot_model_overview(manifest_dict: Dict[str, Any]) -> go.Figure:
    from .parser import manifest_from_dict

    manifest = manifest_from_dict(manifest_dict)

    layer_names = []
    means = []
    stds = []
    sizes = []

    count = 0
    for tensor in manifest.tensors:
        if count >= 50:
            break
        layer_names.append(tensor.name[:30])
        sizes.append(tensor.n_elements / 1e6)
        count += 1

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(range(len(layer_names))),
            y=sizes,
            name="Size (M params)",
            marker_color="rgba(70, 130, 180, 0.8)",
        )
    )

    fig.update_layout(
        title="Model Layer Sizes",
        xaxis_title="Layer Index",
        yaxis_title="Parameters (Millions)",
        template="plotly_white",
        height=400,
    )

    return fig


def get_tensor_quantization_info(tensor) -> str:
    if tensor is None:
        return "Unknown"
    from .constants import QUANTIZED_TYPE_NAMES

    quant_type = tensor.ggml_type_name
    if quant_type in QUANTIZED_TYPE_NAMES:
        bits_map = {
            "Q4_0": 4.5,
            "Q4_1": 4.5,
            "Q5_0": 5.5,
            "Q5_1": 5.5,
            "Q8_0": 8,
            "Q8_1": 8,
            "Q2_K": 2.5,
            "Q3_K": 3.5,
            "Q4_K": 4.5,
            "Q5_K": 5.5,
            "Q6_K": 6.5,
            "Q8_K": 8,
            "IQ2_XXS": 2.125,
            "IQ2_XS": 2.25,
            "IQ3_XS": 3.25,
            "IQ1_S": 1.125,
            "IQ4_XS": 4.25,
            "IQ2_S": 2.125,
        }
        bits = bits_map.get(quant_type, 4)
        return f"{quant_type} (~{bits} bits/weight)"
    return quant_type
