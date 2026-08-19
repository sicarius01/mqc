"""cdqc selftest — 합성 데이터에 전 파이프라인을 돌려 감도/특이도 검증.

각 (실패, 강도) 조합에서 피쳐의 코호트 directed z를 재서:
- "반응해야 할" 칸: 강도에 따라 단조 증가(슬랙 허용) + 최종 강도 z ≥ z_pass
- "반응하면 안 되는" 칸: 전 강도에서 |z| < z_specificity
기대 표는 spec §7.2. 전부 PASS면 exit 0.

캘리브레이션은 베이스라인(injected_failure=="none")으로만 하고, 주입 케이스를
그 기준으로 채점한다 — 케이스 자체 코호트로 정규화하면 주입이 흡수된다.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import io
from .calibrate import calibrate_frames
from .config import Config
from .pipeline import add_hist_emd, extract_features
from .scoring import add_l2_runtime, score_l3
from .synth.generator import generate_dataset

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
    # cd_nm(코호트 대비)은 코호트 자체의 CD 산포(공정 변동)가 1/cosθ 과대와
    # 같은 크기면 원리적으로 못 잡는다 — 시퀀스 내 이웃 대비인 cd_resid가
    # 올바른 검출 경로 (이미지 간 폭 산포에 면역)
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
}

SANDBOX_OVERRIDES = {
    "paths": {"data_dir": "data/synth", "image_dir": "data/synth",
              "cache_dir": "out/selftest/cache"},
    "data": {"coords": {"convention": "xy"}},
    "cohort": {"min_cohort_n": 20},   # 합성 L2 코호트(케이스당 수십 시퀀스)에 맞춤
}


def sandbox_cfg(cfg: Config) -> Config:
    return cfg.with_overrides(SANDBOX_OVERRIDES)


def ensure_synth(cfg: Config, force: bool = False):
    root = cfg.root / "data" / "synth"
    if force or not (root / "records.csv").exists():
        generate_dataset(cfg, root)
    return root


def _median_z(df: pl.DataFrame, feat: str, failure: str, rank: int,
              scope: str, level: str, signed: bool = False) -> float:
    """케이스 표본의 z 중앙값. signed=True면 부호 보존 z(zs_)로 계통 편향 측정.

    특이도(silent) 판정은 계통 반응만 본다 — worse_when="both" 피쳐의 directed
    z(|z|)는 주입이 산포만 키워도 오르므로(예: defocus에서 delta 산포 증가),
    편향 없는 산포 증가를 특이도 위반으로 치지 않는다.
    """
    zcol = f"zs_{feat}" if signed and f"zs_{feat}" in df.columns else f"z_{feat}"
    if zcol not in df.columns:
        return np.nan
    sel = df.filter((pl.col("injected_failure") == failure)
                    & (pl.col("sev_rank") == rank))
    if level == "l3" and scope == "affected":
        sel = sel.filter(pl.col("affected") == 1)
    elif level == "l3" and scope == "unaffected":
        sel = sel.filter(pl.col("affected") == 0)
    z = sel[zcol].to_numpy()
    z = z[np.isfinite(z)]
    return float(np.median(z)) if len(z) else np.nan


def run_selftest(cfg: Config, force: bool = False) -> tuple[str, bool]:
    cfg2 = sandbox_cfg(cfg)
    ensure_synth(cfg2, force=force)
    records = io.load_records(cfg2)
    l3, l2s, l1 = extract_features(cfg2, records=records, force=force)

    truth_cd = records.select(["image_id", "category_id", "cd_index",
                               "injected_failure", "sev_rank", "affected"])
    truth_img = records.group_by("image_id").agg(
        pl.col("injected_failure").first(), pl.col("sev_rank").first(),
        pl.col("injected_strength").first())

    base_ids = truth_img.filter(pl.col("injected_failure") == "none")["image_id"]
    l3b = l3.filter(pl.col("image_id").is_in(base_ids))
    l2b = l2s.filter(pl.col("image_id").is_in(base_ids))
    l1b = l1.filter(pl.col("image_id").is_in(base_ids))
    cs, thr = calibrate_frames(cfg2, l3b, l2b, l1b, write=False)

    l1e = add_hist_emd(l1, cs)
    l3z = cs.apply(l3, "l3", cfg2).join(truth_cd,
                                        on=["image_id", "category_id", "cd_index"],
                                        how="left")
    l3sc = score_l3(l3z, cfg2, thr["t_soft"])
    l2full = add_l2_runtime(l3sc, l2s, cfg2)
    l2z = cs.apply(l2full, "l2", cfg2).join(truth_img, on="image_id", how="left")
    l1z = cs.apply(l1e, "l1", cfg2).join(truth_img, on="image_id", how="left")
    frames = {"l3": l3sc, "l2": l2z, "l1": l1z}

    st = cfg["selftest"]
    z_pass, z_spec = float(st["z_pass"]), float(st["z_specificity"])
    slack = float(st["mono_slack"])
    fringe = bool(cfg["synthetic"]["fringe"])

    ranks_by_failure = {
        key[0]: sorted(part["sev_rank"].unique().to_list())
        for key, part in records.partition_by(["injected_failure"],
                                              as_dict=True).items()
    }

    lines = ["# cdqc selftest — 피쳐 × 주입 실패 감도표",
             f"# config_hash={cfg2.config_hash}  t_soft={thr['t_soft']:.3f}  "
             f"t_seq={thr['t_seq']:.3f}  t_image={thr['t_image']:.3f}",
             f"# 기준: 반응 = 단조증가(슬랙 {slack}) + 최종 z >= {z_pass} / "
             f"침묵 = 전 강도 |z| < {z_spec}", ""]
    n_pass = n_fail = 0

    def eval_cell(failure: str, feat: str, level: str, scope: str,
                  kind: str) -> tuple[str, bool]:
        ranks = ranks_by_failure.get(failure, [])
        zs = [_median_z(frames[level], feat, failure, r, scope, level,
                        signed=(kind == "silent"))
              for r in ranks]
        finite = [z for z in zs if np.isfinite(z)]
        ztxt = " ".join(f"{z:+.1f}" if np.isfinite(z) else "  na" for z in zs)
        if kind == "respond":
            if not finite or not np.isfinite(zs[-1]):
                return f"{ztxt}  (측정 불가)", False
            mono = all(zs[i + 1] >= zs[i] - slack
                       for i in range(len(zs) - 1)
                       if np.isfinite(zs[i]) and np.isfinite(zs[i + 1]))
            ok = mono and zs[-1] >= z_pass
            return f"{ztxt}  mono={'Y' if mono else 'N'} final={zs[-1]:+.1f}", ok
        worst = max((abs(z) for z in finite), default=0.0)
        return f"{ztxt}  max|z|={worst:.1f}", worst < z_spec

    for failure, exp in EXPECTATIONS.items():
        if failure not in ranks_by_failure:
            lines.append(f"[skip] {failure}: 합성 데이터에 케이스 없음")
            continue
        respond = list(exp["respond"])
        if failure == "defocus" and fringe:
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
    report = "\n".join(lines)

    summary_dir = cfg.path("output_dir") / cfg["report"]["summary_dir"]
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "selftest.txt").write_text(report, encoding="utf-8")
    return report, all_pass
