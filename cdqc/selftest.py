"""cdqc selftest — 합성 데이터로 감도/특이도 검증 (개발 전용, 사내 사용과 무관).

**공개 API만 사용**해서 전 파이프라인을 조립한다 — 사내 사용자가 짤 코드와
같은 모양이므로 API 자체의 dogfooding이기도 하다. 실행:

    python -m cdqc.selftest

각 (실패, 강도) 조합에서 피쳐의 코호트 directed z를 재서:
- "반응해야 할" 칸: 강도에 따라 단조 증가(슬랙 허용) + 최종 강도 z ≥ Z_PASS
- "반응하면 안 되는" 칸: 전 강도에서 |z| < Z_SPEC — **계통 편향 기준**
  (worse_when="both"의 directed z는 산포만 커져도 오르므로 부호 z(zs_)로 잰다)
전부 PASS면 exit 0.

캘리브레이션은 베이스라인(injected_failure=="none")으로만 하고 주입 케이스를
그 기준으로 채점한다 — 케이스 자체 코호트로 정규화하면 주입이 흡수된다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .api import (apply_z, cohort_stats, extract_l1, extract_l2, extract_l3,
                  extract_mask_image, extract_mask_l3, hist_emd,
                  threshold_from_quantile, top_feature)
from .params import Params
from .synth.generator import SynthParams, generate_dataset
from .utils import infer_px_nm, to_nm

Z_PASS = 3.0        # "반응해야 할" 피쳐의 최종 강도 z 하한
Z_SPEC = 1.5        # "반응하면 안 되는" 피쳐의 |z| 상한
MONO_SLACK = 0.5    # 단조 증가 판정 허용 슬랙 (z 단위)

# (feature, level, scope). scope: affected | all | unaffected (L3에만 의미)
EXPECTATIONS: dict[str, dict[str, list[tuple[str, str, str]]]] = {
    "defocus": {
        "respond": [("rise_s", "l3", "affected"), ("rise_e", "l3", "affected")],
        "silent": [("delta_s", "l3", "affected"), ("delta_e", "l3", "affected"),
                   ("cd_resid", "l3", "affected")],
    },
    "low_contrast": {
        "respond": [("cnr_s", "l3", "affected"), ("cnr_e", "l3", "affected")],
        "silent": [("rise_s", "l3", "affected"), ("delta_s", "l3", "affected")],
    },
    "noise_up": {
        "respond": [("cnr_s", "l3", "affected"), ("cnr_e", "l3", "affected"),
                    ("noise_sigma", "l1", "all")],
        "silent": [("delta_median_s", "l2", "all")],
    },
    "edge_jump": {
        "respond": [("delta_s", "l3", "affected"), ("s_resid", "l3", "affected"),
                    ("dstep_s", "l3", "affected"), ("cd_resid", "l3", "affected")],
        "silent": [("delta_e", "l3", "affected"), ("e_resid", "l3", "affected"),
                   ("dstep_e", "l3", "affected")],
    },
    "systematic_bias": {
        "respond": [("delta_median_s", "l2", "all")],
        "silent": [("s_resid", "l3", "affected"), ("dstep_s", "l3", "affected")],
    },
    # cd_nm(코호트 대비)은 코호트 CD 산포(공정 변동)가 1/cosθ 과대와 같은
    # 크기면 원리적으로 못 잡는다 — 이웃 대비인 cd_resid가 올바른 검출 경로
    "oblique": {
        "respond": [("obliquity", "l3", "affected"), ("cd_resid", "l3", "affected")],
        "silent": [("delta_s", "l3", "affected"), ("cnr_s", "l3", "affected")],
    },
    "missing": {
        "respond": [("n_cd", "l2", "all")],
        "silent": [("delta_s", "l3", "all"), ("cd_resid", "l3", "all"),
                   ("cnr_s", "l3", "all")],
    },
    "double_edge": {
        "respond": [("margin_s", "l3", "affected"), ("npk_s", "l3", "affected")],
        "silent": [("cnr_s", "l3", "affected")],
    },
    "plateau_defect": {
        "respond": [("plateau_cv", "l3", "affected")],
        "silent": [("delta_s", "l3", "affected")],
    },
    "saturation": {
        "respond": [("sat_lo", "l1", "all")],
        "silent": [],
    },
    "partial_damage": {
        "respond": [("tile_energy_cv", "l1", "all")],
        "silent": [("delta_s", "l3", "unaffected")],
    },
    # ---- 마스크 주입 (변경 #03 §1-d) --------------------------------------
    "mask_shift": {
        "respond": [("mdist_s", "l3", "affected"), ("mdist_e", "l3", "affected"),
                    ("mgrad_s", "l3", "affected"), ("mgrad_e", "l3", "affected"),
                    ("mask_grad_agree", "lm", "all")],
        "silent": [("cnr_s", "l3", "affected"), ("delta_s", "l3", "affected")],
    },
    "mask_ragged": {
        "respond": [("mask_boundary_rough", "lm", "all"),
                    ("mask_n_components", "lm", "all")],
        "silent": [("mdist_s", "l3", "affected")],
    },
    # ---- 총체적 실패 (변경 #03 §2) ----------------------------------------
    "rotated_frame": {
        "respond": [("angle_median", "l2", "all"), ("n_cd", "l2", "all")],
        "silent": [("delta_s", "l3", "affected")],
    },
}

_TRUTH_COLS = ("image_id", "category_id", "cd_index",
               "injected_failure", "sev_rank", "affected")


def build_frames(records: pd.DataFrame, images: dict[str, np.ndarray],
                 masks: dict, params: Params
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """공개 API로 L3/L2/L1/LM 프레임 조립 — 사내 사용자 코드와 같은 모양."""
    records = records.copy()
    records["value_nm"] = to_nm(records["value"].to_numpy(),
                                records["unit"].to_numpy())

    px_nm = {iid: infer_px_nm(g[["sx", "sy"]].to_numpy(),
                              g[["ex", "ey"]].to_numpy(),
                              g["value_nm"].to_numpy())
             for iid, g in records.groupby("image_id", sort=False)}

    parts = []
    lm_rows = []
    for (iid, cat), g in records.groupby(["image_id", "category_id"], sort=False):
        S = g[["sx", "sy"]].to_numpy()
        E = g[["ex", "ey"]].to_numpy()
        f = extract_l3(images[iid], S, E, px_nm[iid],
                       value_nm=g["value_nm"].to_numpy(), params=params)
        # 마스크 정합 피쳐 — extract_l3와 행 순서가 같아 index로 join
        fm = extract_mask_l3(masks[(iid, cat)], images[iid], S, E,
                             px_nm[iid], params=params)
        f = pd.concat([f, fm], axis=1)
        for c in _TRUTH_COLS:
            f[c] = g[c].to_numpy()
        parts.append(f)
        lm_rows.append({"image_id": iid, "category_id": cat,
                        **extract_mask_image(masks[(iid, cat)], images[iid],
                                             params)})
    l3 = pd.concat(parts, ignore_index=True)

    img_truth = (records.groupby("image_id", sort=False)
                 .agg(injected_failure=("injected_failure", "first"),
                      sev_rank=("sev_rank", "first")).reset_index())
    l2 = extract_l2(l3, ["image_id", "category_id"], params).merge(
        img_truth, on="image_id")
    l1 = pd.DataFrame([{"image_id": iid, **extract_l1(img, params)}
                       for iid, img in images.items()]).merge(
        img_truth, on="image_id")
    lm = pd.DataFrame(lm_rows).merge(img_truth, on="image_id")
    return l3, l2, l1, lm


def _z_per_category(df: pd.DataFrame, base: pd.DataFrame,
                    params: Params) -> pd.DataFrame:
    """카테고리별 코호트로 통계 → z (코호트 분리는 호출자가 한다는 API 계약)."""
    parts = []
    for cat, g in df.groupby("category_id", sort=False):
        st = cohort_stats(base[base["category_id"] == cat], params=params)
        parts.append(apply_z(g, st, params))
    return pd.concat(parts).sort_index()


def run_selftest(sp: SynthParams | None = None,
                 params: Params | None = None) -> tuple[str, bool]:
    sp = sp or SynthParams()
    params = params or Params()
    records, images, masks = generate_dataset(sp)
    l3, l2, l1, lm = build_frames(records, images, masks, params)

    base3 = l3[l3["injected_failure"] == "none"]
    z3 = _z_per_category(l3, base3, params)
    z2 = _z_per_category(l2, l2[l2["injected_failure"] == "none"], params)
    zm = _z_per_category(lm, lm[lm["injected_failure"] == "none"], params)

    base1 = l1[l1["injected_failure"] == "none"]
    hists = np.stack([np.asarray(h) for h in base1["hist"]])
    template = np.median(hists, axis=0)
    template = template / template.sum()
    l1 = l1.copy()
    l1["hist_emd"] = [hist_emd(np.asarray(h), template) for h in l1["hist"]]
    z1 = apply_z(l1, cohort_stats(l1[l1["injected_failure"] == "none"],
                                  params=params), params)

    frames = {"l3": z3, "l2": z2, "l1": z1, "lm": zm}
    t_soft = threshold_from_quantile(
        top_feature(z3[z3["injected_failure"] == "none"])["top_z"], 0.90)

    ranks_by_failure = {f: sorted(g["sev_rank"].unique())
                        for f, g in records.groupby("injected_failure")}

    def median_z(level: str, feat: str, failure: str, rank: int,
                 scope: str, signed: bool) -> float:
        df = frames[level]
        col = f"zs_{feat}" if signed and f"zs_{feat}" in df.columns else f"z_{feat}"
        if col not in df.columns:
            return np.nan
        sel = df[(df["injected_failure"] == failure) & (df["sev_rank"] == rank)]
        if level == "l3" and scope == "affected":
            sel = sel[sel["affected"] == 1]
        elif level == "l3" and scope == "unaffected":
            sel = sel[sel["affected"] == 0]
        z = sel[col].to_numpy(dtype=np.float64)
        z = z[np.isfinite(z)]
        return float(np.median(z)) if len(z) else np.nan

    lines = ["# cdqc selftest — 피쳐 × 주입 실패 감도표 (공개 API 경유)",
             f"# 참고 t_soft(베이스라인 top_z p90) = {t_soft:.3f}",
             f"# 기준: 반응 = 단조증가(슬랙 {MONO_SLACK}) + 최종 z >= {Z_PASS} / "
             f"침묵 = 전 강도 |z(부호)| < {Z_SPEC}", ""]
    n_pass = n_fail = 0

    def eval_cell(failure: str, feat: str, level: str, scope: str,
                  kind: str) -> tuple[str, bool]:
        ranks = ranks_by_failure.get(failure, [])
        zs = [median_z(level, feat, failure, r, scope, signed=(kind == "silent"))
              for r in ranks]
        finite = [z for z in zs if np.isfinite(z)]
        ztxt = " ".join(f"{z:+.1f}" if np.isfinite(z) else "  na" for z in zs)
        if kind == "respond":
            if not finite or not np.isfinite(zs[-1]):
                return f"{ztxt}  (측정 불가)", False
            mono = all(zs[i + 1] >= zs[i] - MONO_SLACK
                       for i in range(len(zs) - 1)
                       if np.isfinite(zs[i]) and np.isfinite(zs[i + 1]))
            ok = mono and zs[-1] >= Z_PASS
            return f"{ztxt}  mono={'Y' if mono else 'N'} final={zs[-1]:+.1f}", ok
        worst = max((abs(z) for z in finite), default=0.0)
        return f"{ztxt}  max|z|={worst:.1f}", worst < Z_SPEC

    for failure, exp in EXPECTATIONS.items():
        if failure not in ranks_by_failure:
            lines.append(f"[skip] {failure}: 합성 데이터에 케이스 없음")
            continue
        respond = list(exp["respond"])
        if failure == "defocus" and sp.fringe:
            respond += [("overshoot_s", "l3", "affected"),
                        ("overshoot_e", "l3", "affected")]
        for feat, level, scope in respond:
            txt, ok = eval_cell(failure, feat, level, scope, "respond")
            n_pass, n_fail = n_pass + ok, n_fail + (not ok)
            lines.append(f"[respond] {failure:<16} {feat:<15} {scope:<10} {txt}"
                         f"  {'PASS' if ok else 'FAIL'}")
        for feat, level, scope in exp["silent"]:
            txt, ok = eval_cell(failure, feat, level, scope, "silent")
            n_pass, n_fail = n_pass + ok, n_fail + (not ok)
            lines.append(f"[silent]  {failure:<16} {feat:<15} {scope:<10} {txt}"
                         f"  {'PASS' if ok else 'FAIL'}")
        lines.append("")

    all_pass = n_fail == 0
    lines.append(f"총평: PASS {n_pass}, FAIL {n_fail} → "
                 f"{'PASS' if all_pass else 'FAIL'}")
    return "\n".join(lines), all_pass


if __name__ == "__main__":
    import sys
    report, ok = run_selftest()
    print(report)
    sys.exit(0 if ok else 1)
