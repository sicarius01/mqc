"""internal/report.html — 정적, 오프라인, self-contained (외부 리소스 0).

matplotlib PNG를 base64로 인라인한다. 사내 전용 (nm/좌표/이미지 포함 가능).
"""

from __future__ import annotations

import base64
import html as _html
import io as _io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 글리프 지원 폰트 우선 (없으면 DejaVu로 폴백)
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np
import polars as pl

from ..config import Config

_CSS = """
body { font-family: 'Segoe UI', sans-serif; margin: 24px; color: #222; }
h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: 4px; }
table { border-collapse: collapse; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 3px 8px; text-align: right; }
th { background: #f0f0f0; }
td.l { text-align: left; }
.fail { color: #c22; font-weight: bold; }
.ok { color: #282; }
code { background: #f5f5f5; padding: 1px 4px; }
img.overlay { max-width: 420px; margin: 4px; border: 1px solid #aaa; }
"""


def _fig_b64(fig) -> str:
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img_tag(b64: str, cls: str = "") -> str:
    return f'<img class="{cls}" src="data:image/png;base64,{b64}">'


def _esc(s) -> str:
    return _html.escape(str(s))


def _z_hist_grid(df: pl.DataFrame, title: str) -> str:
    cols = [c for c in df.columns if c.startswith("z_")]
    if not cols:
        return ""
    n = len(cols)
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        x = df[c].to_numpy()
        x = x[np.isfinite(x)]
        if len(x):
            ax.hist(np.clip(x, -6, 8), bins=40, color="#4878a8")
        ax.set_title(c[2:], fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    return _img_tag(_fig_b64(fig))


def _reason_bar(l3: pl.DataFrame) -> str:
    flagged = l3.filter(pl.col("flag"))
    if flagged.height == 0:
        return "<p>플래그된 CD 없음</p>"
    counts = flagged.group_by("reason_code").len().sort("len", descending=True)
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.barh([r for r, _ in counts.rows()][::-1],
            [c for _, c in counts.rows()][::-1], color="#a85050")
    ax.set_title("CD 플래그 사유 분포", fontsize=10)
    ax.tick_params(labelsize=8)
    return _img_tag(_fig_b64(fig))


def write_html(cfg: Config, frames: dict, overlay_dir: Path | None,
               out_path: Path) -> None:
    l3, l2, l1 = frames["l3"], frames["l2"], frames["l1"]
    thr = frames["thresholds"]
    parts = [f"<!doctype html><meta charset='utf-8'><title>cdqc report</title>"
             f"<style>{_CSS}</style>",
             "<h1>cdqc report (internal)</h1>",
             f"<p><code>config {cfg.config_hash}</code> "
             f"<code>calibrated {cfg.calibrated_hash}</code> "
             f"<code>t_soft {thr['t_soft']:.2f}</code> "
             f"<code>t_seq {thr['t_seq']:.2f}</code> "
             f"<code>t_image {thr['t_image']:.2f}</code></p>"]

    n_img, n_fail = l1.height, int(l1["image_fail"].sum())
    n_cd, n_flag = l3.height, int(l3["flag"].sum())
    parts.append("<h2>개요</h2><table>"
                 "<tr><th></th><th>전체</th><th>플래그/FAIL</th><th>비율</th></tr>"
                 f"<tr><td class='l'>이미지</td><td>{n_img}</td><td>{n_fail}</td>"
                 f"<td>{100 * n_fail / max(n_img, 1):.1f}%</td></tr>"
                 f"<tr><td class='l'>CD</td><td>{n_cd}</td><td>{n_flag}</td>"
                 f"<td>{100 * n_flag / max(n_cd, 1):.1f}%</td></tr></table>")
    parts.append(f"<p>G0: {int(l1['g0_hit'].sum())} · G1: {int(l1['g1_hit'].sum())}"
                 f" · G2: {int(l1['g2_hit'].sum())}</p>")

    fails = l1.filter(pl.col("image_fail")).sort("image_id").head(200)
    if fails.height:
        rows = ["<tr><th>image</th><th>recipe</th><th>사유</th>"
                "<th>bad cats</th></tr>"]
        for r in fails.iter_rows(named=True):
            rows.append(f"<tr><td class='l'>{_esc(r['image_id'])}</td>"
                        f"<td class='l'>{_esc(r['recipe_id'])}</td>"
                        f"<td class='l fail'>{_esc(r['image_reasons'])}</td>"
                        f"<td>{r['n_bad_categories']}</td></tr>")
        parts.append("<h2>FAIL 이미지</h2><table>" + "".join(rows) + "</table>")

    worst = (l2.sort("impact_nm", descending=True, nulls_last=True).head(15))
    rows = ["<tr><th>image</th><th>cat</th><th>impact_nm</th><th>tol_nm</th>"
            "<th>frac_flagged</th><th>max_run</th><th>G1</th><th>G2</th></tr>"]
    for r in worst.iter_rows(named=True):
        rows.append(
            f"<tr><td class='l'>{_esc(r['image_id'])}</td><td>{_esc(r['category_id'])}</td>"
            f"<td>{r['impact_nm']:.3f}</td><td>{r['tolerance_nm']:.2f}</td>"
            f"<td>{r['frac_flagged']:.2f}</td><td>{r['max_run']}</td>"
            f"<td>{'X' if r['g1_hit'] else ''}</td><td>{'X' if r['g2_hit'] else ''}</td></tr>")
    parts.append("<h2>impact_nm 상위 시퀀스</h2><table>" + "".join(rows) + "</table>")

    parts.append("<h2>CD 플래그 사유</h2>" + _reason_bar(l3))
    parts.append("<h2>L3 z 분포</h2>" + _z_hist_grid(l3, "L3 directed z"))
    parts.append("<h2>L2/L1 z 분포</h2>" + _z_hist_grid(l2, "L2")
                 + _z_hist_grid(l1, "L1"))

    if overlay_dir is not None and overlay_dir.exists():
        pngs = sorted(overlay_dir.glob("*.png"))[:12]
        if pngs:
            parts.append("<h2>오버레이 (상위 12)</h2>")
            for p in pngs:
                b64 = base64.b64encode(p.read_bytes()).decode()
                parts.append(f"<div><b>{_esc(p.stem)}</b><br>"
                             + _img_tag(b64, "overlay") + "</div>")

    out_path.write_text("\n".join(parts), encoding="utf-8")
