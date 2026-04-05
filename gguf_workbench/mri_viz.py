from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DARK_BG = "#0a0e17"
DARK_PANEL = "#111827"
GRID_COLOR = "#1e293b"
TEXT_COLOR = "#e2e8f0"
MUTED_TEXT = "#94a3b8"
ACCENT_BLUE = "#3b82f6"
ACCENT_CYAN = "#06b6d4"
ACCENT_GREEN = "#10b981"
ACCENT_AMBER = "#f59e0b"
ACCENT_RED = "#ef4444"
ACCENT_PURPLE = "#8b5cf6"


def compute_tensor_stats(arr: np.ndarray) -> Dict[str, Any]:
    flat = arr.reshape(-1).astype(np.float64)
    abs_flat = np.abs(flat)
    sorted_abs = np.sort(abs_flat)
    n = flat.size

    percentiles = {
        "p1": float(np.percentile(flat, 1)),
        "p5": float(np.percentile(flat, 5)),
        "p25": float(np.percentile(flat, 25)),
        "p50": float(np.percentile(flat, 50)),
        "p75": float(np.percentile(flat, 75)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
    }

    mean_val = float(flat.mean())
    std_val = float(flat.std())
    iqr = percentiles["p75"] - percentiles["p25"]
    lower_fence = percentiles["p25"] - 1.5 * iqr
    upper_fence = percentiles["p75"] + 1.5 * iqr
    outlier_mask = (flat < lower_fence) | (flat > upper_fence)
    n_outliers = int(outlier_mask.sum())

    if std_val > 0:
        skewness = float(((flat - mean_val) ** 3).mean() / std_val**3)
        kurtosis = float(((flat - mean_val) ** 4).mean() / std_val**4 - 3)
    else:
        skewness = 0.0
        kurtosis = 0.0

    n_zeros = int((flat == 0).sum())
    sparsity = n_zeros / n if n > 0 else 0.0

    sign_changes = np.diff(np.signbit(flat))
    zero_crossings = int(sign_changes.sum())
    zero_crossing_rate = zero_crossings / (n - 1) if n > 1 else 0.0

    energy = float((flat**2).sum())
    rms = float(np.sqrt((flat**2).mean())) if n > 0 else 0.0

    top_100_sum = float(sorted_abs[-min(100, n) :].sum()) if n > 0 else 0.0
    total_abs_sum = float(sorted_abs.sum())
    top_100_concentration = top_100_sum / total_abs_sum if total_abs_sum > 0 else 0.0

    return {
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": mean_val,
        "std": std_val,
        "n_elements": n,
        "shape": list(arr.shape),
        "percentiles": percentiles,
        "iqr": float(iqr),
        "n_outliers": n_outliers,
        "outlier_pct": n_outliers / n * 100 if n > 0 else 0.0,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "n_zeros": n_zeros,
        "sparsity": sparsity,
        "zero_crossings": zero_crossings,
        "zero_crossing_rate": zero_crossing_rate,
        "energy": energy,
        "rms": rms,
        "top_100_concentration": top_100_concentration,
    }


def _dark_layout(fig: go.Figure, title: str, height: int = 400) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=14)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=MUTED_TEXT, size=11),
        height=height,
        hovermode="x unified",
        margin=dict(l=50, r=20, t=40, b=40),
    )
    fig.update_xaxes(
        gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT)
    )
    fig.update_yaxes(
        gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT)
    )
    return fig


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
            marker_color="rgba(59, 130, 246, 0.65)",
            marker_line_color="rgba(59, 130, 246, 0.3)",
            marker_line_width=0.5,
            opacity=0.9,
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
                    line=dict(color=ACCENT_CYAN, width=2.5),
                    fill="tozeroy",
                    fillcolor="rgba(6, 182, 212, 0.08)",
                )
            )
        except Exception:
            pass

    mean_val = float(flat.mean())
    std_val = float(flat.std())

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=ACCENT_GREEN,
        line_width=2,
        annotation_text="μ",
    )
    fig.add_vline(
        x=mean_val + std_val,
        line_dash="dot",
        line_color=ACCENT_AMBER,
        line_width=1.5,
        annotation_text="+1σ",
    )
    fig.add_vline(
        x=mean_val - std_val,
        line_dash="dot",
        line_color=ACCENT_AMBER,
        line_width=1.5,
        annotation_text="-1σ",
    )
    fig.add_vline(
        x=mean_val + 2 * std_val,
        line_dash="dot",
        line_color=ACCENT_RED,
        line_width=1,
        annotation_text="+2σ",
    )
    fig.add_vline(
        x=mean_val - 2 * std_val,
        line_dash="dot",
        line_color=ACCENT_RED,
        line_width=1,
        annotation_text="-2σ",
    )

    if show_stats:
        stats_text = (
            f"<b style='color:{TEXT_COLOR}'>Distribution Stats</b><br>"
            f"Min: {flat.min():.4f}<br>"
            f"Max: {flat.max():.4f}<br>"
            f"Mean: {mean_val:.4f}<br>"
            f"Std: {std_val:.4f}<br>"
            f"Median: {float(np.median(flat)):.4f}"
        )
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=10, color=TEXT_COLOR),
            bgcolor=DARK_PANEL,
            bordercolor=GRID_COLOR,
            borderwidth=1,
            align="left",
        )

    _dark_layout(fig, title)
    return fig


def plot_magnitude_spectrum(
    arr: np.ndarray,
    title: str = "Weight Magnitude Spectrum",
    subsample: Optional[int] = None,
) -> go.Figure:
    flat = np.abs(arr.reshape(-1).astype(np.float64))
    sorted_vals = np.sort(flat)[::-1]

    if subsample is not None and sorted_vals.shape[0] > subsample:
        indices = np.linspace(0, sorted_vals.shape[0] - 1, subsample, dtype=int)
        sorted_vals = sorted_vals[indices]

    n = len(sorted_vals)
    x = np.arange(n)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=sorted_vals,
            mode="lines",
            name="Magnitude",
            line=dict(color=ACCENT_PURPLE, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.1)",
        )
    )

    top_1pct_idx = max(0, int(n * 0.01) - 1)
    fig.add_hline(
        y=sorted_vals[top_1pct_idx],
        line_dash="dash",
        line_color=ACCENT_AMBER,
        line_width=1,
        annotation_text=f"Top 1% threshold: {sorted_vals[top_1pct_idx]:.4f}",
    )

    fig.update_xaxes(title="Rank (sorted by magnitude)", type="log")
    fig.update_yaxes(title="|Weight|")
    _dark_layout(fig, title)
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

    fig.update_xaxes(title="Column")
    fig.update_yaxes(title="Row")
    _dark_layout(fig, title, height=500)
    return fig


def plot_outlier_heatmap(
    arr: np.ndarray,
    title: str = "Outlier Detection Heatmap",
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

    flat = arr.reshape(-1).astype(np.float64)
    q1 = float(np.percentile(flat, 25))
    q3 = float(np.percentile(flat, 75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outlier_mask = ((arr < lower) | (arr > upper)).astype(np.float32)

    z_values = np.where(outlier_mask > 0, arr, np.nan)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=arr,
            colorscale="RdBu_r",
            hovertemplate="row: %{y}<br>col: %{x}<br>value: %{z:.4f}<extra></extra>",
            opacity=0.3,
            showscale=False,
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=z_values,
            colorscale=[[0, ACCENT_RED], [1, ACCENT_RED]],
            hovertemplate="row: %{y}<br>col: %{x}<br>outlier: %{z:.4f}<extra></extra>",
            showscale=True,
            colorbar=dict(
                title="Outliers",
                tickfont=dict(color=MUTED_TEXT),
                titlefont=dict(color=TEXT_COLOR),
            ),
        )
    )

    n_outliers = int(outlier_mask.sum())
    total = outlier_mask.size
    pct = n_outliers / total * 100 if total > 0 else 0

    annotation_text = (
        f"<b style='color:{ACCENT_RED}'>Outlier Map</b><br>"
        f"Method: IQR (1.5×)<br>"
        f"Range: [{lower:.4f}, {upper:.4f}]<br>"
        f"Outliers: {n_outliers:,} ({pct:.2f}%)"
    )
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        font=dict(size=10, color=TEXT_COLOR),
        bgcolor=DARK_PANEL,
        bordercolor=GRID_COLOR,
        borderwidth=1,
        align="left",
    )

    fig.update_xaxes(title="Column")
    fig.update_yaxes(title="Row")
    _dark_layout(fig, title, height=500)
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

    types = list(set(d["type"] for d in layer_data))
    type_counts = {t: sum(1 for d in layer_data if d["type"] == t) for t in types}

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Tensor Count by Type", "Top 30 Tensor Sizes"),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
    )

    fig.add_trace(
        go.Pie(
            labels=list(type_counts.keys()),
            values=list(type_counts.values()),
            hole=0.4,
            marker=dict(
                colors=[
                    ACCENT_BLUE,
                    ACCENT_CYAN,
                    ACCENT_GREEN,
                    ACCENT_AMBER,
                    ACCENT_RED,
                    ACCENT_PURPLE,
                ]
            ),
            textfont=dict(color=TEXT_COLOR),
        ),
        row=1,
        col=1,
    )

    names = [
        d["name"][:40] + "..." if len(d["name"]) > 40 else d["name"]
        for d in layer_data[:30]
    ]
    sizes = [d["bytes"] / (1024 * 1024) for d in layer_data[:30]]
    fig.add_trace(
        go.Bar(
            x=names,
            y=sizes,
            marker_color=ACCENT_BLUE,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=dict(text="Model Tensor Overview", font=dict(color=TEXT_COLOR)),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=MUTED_TEXT),
        height=400,
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=40),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT))
    fig.update_yaxes(gridcolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT))

    return fig, layer_data


def plot_model_overview(manifest_dict: Dict[str, Any]) -> go.Figure:
    from .parser import manifest_from_dict

    manifest = manifest_from_dict(manifest_dict)

    layer_names = []
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
            marker_color="rgba(59, 130, 246, 0.7)",
            marker_line_color=ACCENT_BLUE,
            marker_line_width=0.5,
        )
    )

    fig.update_xaxes(
        title="Layer Index", gridcolor=GRID_COLOR, tickfont=dict(color=MUTED_TEXT)
    )
    fig.update_yaxes(
        title="Parameters (Millions)",
        gridcolor=GRID_COLOR,
        tickfont=dict(color=MUTED_TEXT),
    )
    _dark_layout(fig, "Model Layer Sizes")
    return fig


def get_tensor_quantization_info(tensor) -> str:
    if tensor is None:
        return "Unknown"
    from .constants import QUANTIZED_TYPE_NAMES

    quant_type = tensor.ggml_type_name
    if quant_type in QUANTIZED_TYPE_NAMES:
        bits_map = {
            "Q4_0": 4.0,
            "Q4_1": 4.5,
            "Q5_0": 5.0,
            "Q5_1": 5.5,
            "Q8_0": 8.0,
            "Q8_1": 8.5,
            "Q2_K": 2.56,
            "Q3_K": 3.44,
            "Q4_K": 4.5,
            "Q5_K": 5.5,
            "Q6_K": 6.5,
            "Q8_K": 8.0,
            "IQ2_XXS": 2.06,
            "IQ2_XS": 2.5,
            "IQ3_XS": 3.44,
            "IQ1_S": 1.1,
            "IQ4_XS": 4.5,
            "IQ2_S": 2.5,
        }
        bits = bits_map.get(quant_type, 4)
        return f"{quant_type} (~{bits} bits/weight)"
    return quant_type


def format_stats_markdown(
    stats: Dict[str, Any], quant_info: str, tensor_name: str
) -> str:
    p = stats["percentiles"]
    sparsity_pct = stats["sparsity"] * 100

    skew_label = "Symmetric"
    if stats["skewness"] > 0.5:
        skew_label = "Right-skewed"
    elif stats["skewness"] < -0.5:
        skew_label = "Left-skewed"

    kurt_label = "Mesokurtic (normal tails)"
    if stats["kurtosis"] > 1:
        kurt_label = "Leptokurtic (heavy tails)"
    elif stats["kurtosis"] < -1:
        kurt_label = "Platykurtic (light tails)"

    return (
        f"### `{tensor_name}`\n"
        f"**Type:** `{quant_info}`  \n"
        f"**Shape:** `{stats['shape']}`  \n"
        f"**Elements:** `{stats['n_elements']:,}`\n\n"
        f"---\n"
        f"#### Central Tendency\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Mean | `{stats['mean']:.6g}` |\n"
        f"| Median | `{p['p50']:.6g}` |\n"
        f"| Std Dev | `{stats['std']:.6g}` |\n"
        f"| RMS | `{stats['rms']:.6g}` |\n"
        f"| Energy | `{stats['energy']:.4e}` |\n\n"
        f"---\n"
        f"#### Percentiles\n"
        f"| P1 | P5 | P25 | P50 | P75 | P95 | P99 |\n"
        f"|----|----|-----|-----|-----|-----|-----|\n"
        f"| `{p['p1']:.4g}` | `{p['p5']:.4g}` | `{p['p25']:.4g}` | `{p['p50']:.4g}` | `{p['p75']:.4g}` | `{p['p95']:.4g}` | `{p['p99']:.4g}` |\n\n"
        f"---\n"
        f"#### Distribution Shape\n"
        f"- **Skewness:** `{stats['skewness']:.4f}` ({skew_label})\n"
        f"- **Kurtosis (excess):** `{stats['kurtosis']:.4f}` ({kurt_label})\n"
        f"- **IQR:** `{stats['iqr']:.6g}`\n\n"
        f"---\n"
        f"#### Sparsity & Outliers\n"
        f"- **Zeros:** `{stats['n_zeros']:,}` ({sparsity_pct:.2f}%)\n"
        f"- **Outliers (IQR):** `{stats['n_outliers']:,}` ({stats['outlier_pct']:.2f}%)\n"
        f"- **Zero-crossing rate:** `{stats['zero_crossing_rate']:.4f}`\n\n"
        f"---\n"
        f"#### Concentration\n"
        f"- **Top 100 weights contribution:** `{stats['top_100_concentration'] * 100:.2f}%` of total magnitude"
    )
