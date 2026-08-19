import math

import pytest

from cdqc.config import (DEFAULTS, dict_hash, dump_toml, load_config,
                         parse_set_overrides)
from cdqc.errors import CdqcError
import tomllib


def test_defaults_load():
    cfg = load_config(None)
    assert cfg["sampling"]["win_frac"] == 0.4
    assert cfg["thresholds"]["t_soft"] == "auto"


def test_set_overrides_types():
    d = parse_set_overrides(["thresholds.t_soft=3.0", "gates.enable_g0=false",
                             "project.name=abc", "cohort.min_cohort_n=50"])
    assert d["thresholds"]["t_soft"] == 3.0
    assert d["gates"]["enable_g0"] is False
    assert d["project"]["name"] == "abc"
    assert d["cohort"]["min_cohort_n"] == 50


def test_set_bad_syntax():
    with pytest.raises(CdqcError) as e:
        parse_set_overrides(["nonsense"])
    assert e.value.code == "E-CONF-04"


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[sampling]\nwin_fraq = 0.5\n", encoding="utf-8")
    with pytest.raises(CdqcError) as e:
        load_config(p)
    assert e.value.code == "E-CONF-02"


def test_wildcard_sections_allowed(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("[thresholds.tolerance_nm]\ndefault = 1.0\nMYCAT = 0.7\n",
                 encoding="utf-8")
    cfg = load_config(p)
    assert cfg.tolerance_nm("MYCAT") == 0.7
    assert cfg.tolerance_nm("OTHER") == 1.0


def test_threshold_auto_without_calibrated(tmp_path):
    cfg = load_config(None, root=str(tmp_path))
    with pytest.raises(CdqcError) as e:
        cfg.threshold("t_soft")
    assert e.value.code == "E-CONF-05"


def test_threshold_fixed_overrides_calibrated():
    cfg = load_config(None, set_overrides=["thresholds.t_soft=2.5"])
    assert cfg.threshold("t_soft") == 2.5


def test_dump_toml_roundtrip():
    import numpy as np
    doc = {
        "a": {"x": 1, "y": 1.5, "z": True, "s": "한글", "nanv": float("nan"),
              "npf": np.float64(2.5), "npi": np.int64(7),
              "lst": [1.0, 2.0, np.float64(3.0)]},
        "tbl": {"r1|A": {"f": [0.1, 0.2], "__n__": 30}},
    }
    text = dump_toml(doc)
    back = tomllib.loads(text)
    assert back["a"]["x"] == 1 and back["a"]["npf"] == 2.5
    assert back["a"]["s"] == "한글"
    assert math.isnan(back["a"]["nanv"])
    assert back["tbl"]["r1|A"]["__n__"] == 30


def test_with_overrides_deep():
    cfg = load_config(None)
    cfg2 = cfg.with_overrides({"cohort": {"min_cohort_n": 20}})
    assert cfg2["cohort"]["min_cohort_n"] == 20
    assert cfg["cohort"]["min_cohort_n"] == DEFAULTS["cohort"]["min_cohort_n"]
    assert cfg2["cohort"]["trim_frac"] == cfg["cohort"]["trim_frac"]


def test_hash_stable():
    cfg1 = load_config(None)
    cfg2 = load_config(None)
    assert cfg1.config_hash == cfg2.config_hash
    cfg3 = load_config(None, set_overrides=["project.seed=43"])
    assert cfg3.config_hash != cfg1.config_hash
