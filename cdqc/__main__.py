"""cdqc CLI.

cdqc <subcommand> [--config config.toml] [--root PATH] [--set key=value ...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Config, load_config, parse_set_overrides
from .errors import CdqcError

log = logging.getLogger("cdqc")


def _setup_logging(cfg: Config) -> None:
    level = getattr(logging, str(cfg["logging"]["level"]).upper(), logging.INFO)
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    logfile = Path(cfg["logging"]["file"])
    if not logfile.is_absolute():
        logfile = cfg.root / logfile
    try:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError:
        pass


def _write_run_outputs(cfg: Config, res: dict) -> None:
    import polars as pl
    from .report.html import write_html
    from .report.overlay import draw_overlays
    from .report.summary import write_summaries

    out = cfg.path("output_dir")
    internal = out / cfg["report"]["internal_dir"]
    internal.mkdir(parents=True, exist_ok=True)
    res["l3"].write_parquet(internal / "per_cd.parquet")
    res["l2"].write_parquet(internal / "per_seq.parquet")
    per_image = res["l1"].drop("hist") if "hist" in res["l1"].columns else res["l1"]
    per_image.write_parquet(internal / "per_image.parquet")

    overlay_dir = internal / "overlay"
    n_ov = draw_overlays(cfg, res["l3"], res["l1"], overlay_dir)
    if cfg["report"]["html"]:
        write_html(cfg, res, overlay_dir, internal / "report.html")
    write_summaries(cfg, res)

    n_img = res["l1"].height
    n_fail = int(res["l1"]["image_fail"].sum())
    n_cd = res["l3"].height
    n_flag = int(res["l3"]["flag"].sum())
    print(f"이미지 {n_img}개 중 FAIL {n_fail} "
          f"({100 * n_fail / max(n_img, 1):.1f}%), "
          f"CD {n_cd}개 중 플래그 {n_flag} ({100 * n_flag / max(n_cd, 1):.1f}%)")
    print(f"임계값: t_soft={res['thresholds']['t_soft']:.2f} "
          f"t_seq={res['thresholds']['t_seq']:.2f} "
          f"t_image={res['thresholds']['t_image']:.2f}")
    print(f"출력: {internal}  (오버레이 {n_ov}개) / {out / cfg['report']['summary_dir']}")


def cmd_doctor(cfg: Config, args) -> int:
    from .doctor import run_doctor
    report, ok = run_doctor(cfg)
    print(report)
    return 0 if ok else 1


def cmd_synth(cfg: Config, args) -> int:
    from .synth.generator import generate_dataset
    out = cfg.root / "data" / "synth"
    df = generate_dataset(cfg, out)
    print(f"합성 데이터 생성: {out}  레코드 {df.height}건, "
          f"이미지 {df['image_id'].n_unique()}개")
    return 0


def cmd_selftest(cfg: Config, args) -> int:
    from .selftest import run_selftest
    report, all_pass = run_selftest(cfg, force=args.force)
    print(report)
    return 0 if all_pass else 1


def cmd_extract(cfg: Config, args) -> int:
    from .pipeline import extract_features
    l3, l2, l1 = extract_features(cfg, force=args.force)
    print(f"피쳐 캐시: L3 {l3.height}행, L2 {l2.height}행, L1 {l1.height}행 "
          f"→ {cfg.path('cache_dir')}")
    return 0


def cmd_calibrate(cfg: Config, args) -> int:
    from .calibrate import run_calibrate
    thr = run_calibrate(cfg, force=args.force, baseline=args.baseline)
    print(f"calibrated.toml 작성: {cfg.path('calibrated')}")
    print(f"auto 임계값: {thr}")
    if args.baseline is None:
        print("주의: auto 임계값은 '대부분 정상' 가정의 분위수 — 불량률이 높은 "
              "데이터셋이면 --baseline <정상 image_id 목록 파일>을 쓸 것")
    return 0


def cmd_run(cfg: Config, args) -> int:
    from .pipeline import run_pipeline
    if args.sweep:
        key, _, raw = args.sweep.partition("=")
        if not raw:
            raise CdqcError("E-CONF-04", f"--sweep {args.sweep}")
        values = [v.strip() for v in raw.split(",")]
        lines = [f"# sweep {key}"]
        for v in values:
            cfg2 = cfg.with_overrides(parse_set_overrides([f"{key}={v}"]))
            res = run_pipeline(cfg2, force=args.force)
            sub = cfg.path("output_dir") / "sweep" / f"{key.replace('.', '_')}={v}"
            sub.mkdir(parents=True, exist_ok=True)
            res["l1"].drop("hist").write_parquet(sub / "per_image.parquet")
            n_img, n_fail = res["l1"].height, int(res["l1"]["image_fail"].sum())
            n_cd, n_flag = res["l3"].height, int(res["l3"]["flag"].sum())
            line = (f"{key}={v:<8} CD플래그 {100 * n_flag / max(n_cd, 1):5.1f}%  "
                    f"이미지FAIL {100 * n_fail / max(n_img, 1):5.1f}%")
            lines.append(line)
            print(line)
        sdir = cfg.path("output_dir") / cfg["report"]["summary_dir"]
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "sweep.txt").write_text("\n".join(lines), encoding="utf-8")
        return 0
    res = run_pipeline(cfg, force=args.force)
    _write_run_outputs(cfg, res)
    return 0


def cmd_evaluate(cfg: Config, args) -> int:
    from .evaluate import run_evaluate
    print(run_evaluate(cfg))
    return 0


def cmd_explore(cfg: Config, args) -> int:
    from .explore import run_explore
    print(run_explore(cfg))
    return 0


COMMANDS = {
    "doctor": cmd_doctor, "synth": cmd_synth, "selftest": cmd_selftest,
    "extract": cmd_extract, "calibrate": cmd_calibrate, "run": cmd_run,
    "evaluate": cmd_evaluate, "explore": cmd_explore,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cdqc",
                                 description="TEM CD 측정 품질 판정 툴")
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--config", default=None, help="config.toml 경로 (기본: ./config.toml)")
    ap.add_argument("--root", default=None, help="[project].root 오버라이드")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="TOML 경로 오버라이드 (예: thresholds.t_soft=3.0)")
    ap.add_argument("--force", action="store_true",
                    help="캐시/합성 데이터 무시하고 재생성")
    ap.add_argument("--sweep", default=None, metavar="KEY=V1,V2,...",
                    help="run 전용: 설정값 스윕")
    ap.add_argument("--baseline", default=None, metavar="PATH",
                    help="calibrate 전용: 정상 image_id 목록 파일 (한 줄에 하나)")
    args = ap.parse_args(argv)

    config_path = args.config
    if config_path is None and Path("config.toml").exists():
        config_path = "config.toml"

    try:
        cfg = load_config(config_path, root=args.root, set_overrides=args.set)
        _setup_logging(cfg)
        return COMMANDS[args.command](cfg, args)
    except CdqcError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
