"""Local Plotly figures; coordinates always remain original image pixels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def image_overlay(dataset, rows: pd.DataFrame, selected: pd.Series | None,
                  show_mask=False, focus=False, contrast=(0, 255)):
    if selected is not None and not np.isfinite([selected.s_x, selected.s_y, selected.e_x, selected.e_y]).all():
        selected = None
        focus = False
    fig = go.Figure()
    img = dataset.img
    if img is not None:
        step = max(1, int(np.ceil(max(img.shape) / 1000)))
        x0, y0, x1, y1 = 0, 0, img.shape[1], img.shape[0]
        if focus and selected is not None and np.isfinite([selected.s_x, selected.s_y, selected.e_x, selected.e_y]).all():
            pad = max(15, np.hypot(selected.e_x-selected.s_x, selected.e_y-selected.s_y)*.5)
            x0, x1 = max(0, int(min(selected.s_x, selected.e_x)-pad)), min(img.shape[1], int(max(selected.s_x, selected.e_x)+pad)+1)
            y0, y1 = max(0, int(min(selected.s_y, selected.e_y)-pad)), min(img.shape[0], int(max(selected.s_y, selected.e_y)+pad)+1)
            step = 1  # Pixel-level offset inspection must never use a downsampled overview.
        fig.add_trace(go.Heatmap(z=img[y0:y1:step, x0:x1:step],
                                x=np.arange(x0, x1, step),
                                y=np.arange(y0, y1, step),
                                colorscale="Gray", zmin=contrast[0], zmax=contrast[1],
                                showscale=False, hovertemplate="x=%{x}px · y=%{y}px<br>명암=%{z}<extra></extra>"))
        if show_mask and dataset.labelmap is not None:
            fig.add_trace(go.Heatmap(z=dataset.labelmap[y0:y1:step, x0:x1:step],
                                    x=np.arange(x0, x1, step),
                                    y=np.arange(y0, y1, step),
                                    colorscale="Turbo", opacity=.28, showscale=False,
                                    hovertemplate="라벨=%{z}<extra></extra>"))
    if not rows.empty:
        x, y = [], []
        for r in rows.itertuples():
            x.extend([r.s_x, r.e_x, None])
            y.extend([r.s_y, r.e_y, None])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#22c8bc", width=1.5),
                                name="같은 카테고리 CD", hoverinfo="skip"))
    if selected is not None:
        fig.add_trace(go.Scatter(x=[selected.s_x, selected.e_x], y=[selected.s_y, selected.e_y],
                                mode="lines+markers+text", text=["S", "E"],
                                textposition="top center", textfont=dict(color="#ffcc4d", size=16),
                                line=dict(color="#ffcc4d", width=4), marker=dict(size=10),
                                name="선택 CD", hovertemplate="%{text}: (%{x:.3f}, %{y:.3f})px<extra></extra>"))
        if focus:
            lo_x, hi_x = min(selected.s_x, selected.e_x), max(selected.s_x, selected.e_x)
            lo_y, hi_y = min(selected.s_y, selected.e_y), max(selected.s_y, selected.e_y)
            margin = max(15, (hi_x - lo_x + hi_y - lo_y) * .5)
            fig.update_xaxes(range=[lo_x - margin, hi_x + margin])
            fig.update_yaxes(range=[hi_y + margin, lo_y - margin])
    fig.update_layout(height=550, margin=dict(l=10, r=10, t=10, b=10),
                      dragmode="pan", legend=dict(orientation="h"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#142833")
    fig.update_xaxes(title="x · col (px)", constrain="domain")
    fig.update_yaxes(title="y · row (px)", autorange=False if focus and selected is not None else "reversed",
                     scaleanchor="x", scaleratio=1)
    return fig


def profile_plot(table: pd.DataFrame, selected=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.12,
                        subplot_titles=("리본 평균 명암", "그래디언트"))
    x = table["s_px"]
    for name, row, color in [("intensity", 1, "#087f8c"), ("gradient", 2, "#db7c26")]:
        if name in table:
            fig.add_trace(go.Scatter(x=x, y=table[name], mode="lines", name=name,
                                    line=dict(color=color)), row=row, col=1)
    attrs = table.attrs
    endpoints = [("보고 S", attrs.get("s_endpoint_px", 0.0)),
                 ("보고 E", attrs.get("e_endpoint_px", attrs.get("length_px")))]
    for name, xpos in endpoints:
        if xpos is not None:
            fig.add_vline(x=float(xpos), line_dash="dash", line_color="#c45835",
                          annotation_text=name, row=1, col=1)
            fig.add_vline(x=float(xpos), line_dash="dash", line_color="#c45835", row=2, col=1)
    for _, endpoint in endpoints:
        half = attrs.get("window_half_px")
        window = (endpoint-half, endpoint+half) if endpoint is not None and half is not None else None
        if window is not None and len(window) == 2:
            for row in (1, 2):
                fig.add_vrect(x0=window[0], x1=window[1], fillcolor="#73c9be", opacity=.12,
                              line_width=0, row=row, col=1)
    if selected is not None and float(selected.get("px_nm", 0)) > 0:
        for name, endpoint, delta in [("S 피크", endpoints[0][1], selected.get("delta_s")),
                                      ("E 피크", endpoints[1][1], selected.get("delta_e"))]:
            if endpoint is not None and delta is not None and np.isfinite(delta):
                position = endpoint + float(delta) / float(selected["px_nm"])
                fig.add_vline(x=position, line_color="#7262b0", line_dash="dot",
                              annotation_text=name, row=2, col=1)
    fig.update_xaxes(title="보고 S에서 S→E 방향 거리 (px)", row=2, col=1)
    fig.update_layout(height=510, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    return fig
