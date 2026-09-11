"""Remembered, explicit file matching for a local workbench directory.

Discovery reads filenames only. It never guesses between duplicate candidates,
and retains unmatched files so matching rules can be diagnosed in the GUI.
"""

from __future__ import annotations

import copy
import fnmatch
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile

import pandas as pd


DEFAULT_CONFIG_PATH = Path("out/workbench/config.json")
ROLES = ("image", "table", "seg", "dm3")
_SKIP_DIRS = {"out", ".git", ".venv", "venv", "__pycache__", "node_modules"}
_COLUMNS = ["image_id", *(f"{role}_path" for role in ROLES), "px_nm",
            "status", "issues", "enabled", *(f"{role}_candidates" for role in ROLES),
            "match_details"]


def default_config() -> dict:
    """A new independent settings dictionary; no filesystem access."""
    return {
        "version": 1, "root": "", "recursive": True,
        "patterns": {"image": "*.tif;*.tiff", "table": "*.xlsx;*.csv",
                     "seg": "*.png", "dm3": "*.dm3"},
        "roles": {
            "dm3": {"directory": None, "extension": ".dm3", "suffix": ""},
            "image": {"directory": None, "extension": ".tif", "suffix": ""},
            "seg": {"directory": "result", "extension": ".png", "suffix": ""},
            "table": {"directory": "result", "extension": ".xlsx", "suffix": "_Result"},
        },
        "match_mode": "layout", "key_source": "stem", "key_regex": "",
        "role_key_regex": {}, "require_seg": True, "require_dm3": True,
        "px_nm": None, "ignored_patterns": [], "options": {"seg_restore": True},
    }


def _patterns(value, field: str) -> list[str]:
    if isinstance(value, str):
        values = value.split(";")
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        values = value
    else:
        raise ValueError(f"{field}: use a semicolon-separated string or a list of strings")
    return [v.strip().replace("\\", "/") for v in values if v.strip()]


def validate_config(config: dict) -> dict:
    """Validate and merge defaults, retaining additional UI/backend options."""
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    out = default_config()
    out.update(copy.deepcopy(config))
    if out["version"] != 1:
        raise ValueError(f"Unsupported configuration version: {out['version']!r}")
    if not isinstance(out["root"], str):
        raise ValueError("root must be a directory path string")
    for name in ("recursive", "require_seg", "require_dm3"):
        if not isinstance(out[name], bool):
            raise ValueError(f"{name} must be true or false")
    if out["match_mode"] not in {"layout", "stem", "folder", "regex"}:
        raise ValueError("match_mode must be layout, stem, folder, or regex")
    supplied_roles = out["roles"]
    if not isinstance(supplied_roles, dict) or set(supplied_roles) - set(ROLES):
        raise ValueError(f"roles must be an object with roles: {', '.join(ROLES)}")
    out["roles"] = default_config()["roles"]
    for role in ROLES:
        values = supplied_roles.get(role, {})
        if not isinstance(values, dict):
            raise ValueError(f"roles.{role} must contain directory, extension, and suffix")
        out["roles"][role].update(values)
        rule = out["roles"][role]
        directory = rule["directory"]
        if directory is not None:
            if not isinstance(directory, str):
                raise ValueError(f"roles.{role}.directory must be a relative directory or null")
            directory = directory.strip().replace("\\", "/")
            if (PurePosixPath(directory).is_absolute() or PureWindowsPath(directory).drive
                    or ".." in directory.split("/")):
                raise ValueError(f"roles.{role}.directory must be relative, without '..' or an absolute path")
            rule["directory"] = str(PurePosixPath(directory)) if directory else ""
            if rule["directory"] == ".":
                rule["directory"] = ""
        extension = rule["extension"]
        if not isinstance(extension, str) or not extension.strip():
            raise ValueError(f"roles.{role}.extension must be a single file extension")
        extension = "." + extension.strip().lstrip(".")
        if extension == "." or any(char in extension[1:] for char in ".\\/*?[]"):
            raise ValueError(f"roles.{role}.extension must be a single literal file extension")
        rule["extension"] = extension
        if not isinstance(rule["suffix"], str) or any(char in rule["suffix"] for char in "\\/"):
            raise ValueError(f"roles.{role}.suffix must be literal filename text without directory separators")
    if out["key_source"] not in {"stem", "relative_path"}:
        raise ValueError("key_source must be stem or relative_path")
    supplied_patterns = out["patterns"]
    if not isinstance(supplied_patterns, dict) or set(supplied_patterns) - set(ROLES):
        raise ValueError(f"patterns must be an object with roles: {', '.join(ROLES)}")
    out["patterns"] = {**default_config()["patterns"], **supplied_patterns}
    for role, value in out["patterns"].items():
        if not _patterns(value, f"patterns.{role}"):
            raise ValueError(f"patterns.{role} must contain at least one file pattern")
    _patterns(out["ignored_patterns"], "ignored_patterns")
    if not isinstance(out["role_key_regex"], dict) or set(out["role_key_regex"]) - set(ROLES):
        raise ValueError(f"role_key_regex must be an object with roles: {', '.join(ROLES)}")
    regexes = {"key_regex": out["key_regex"], **{
        f"role_key_regex.{role}": regex for role, regex in out["role_key_regex"].items()}}
    for field, regex in regexes.items():
        if not isinstance(regex, str):
            raise ValueError(f"{field} must be a regular expression string")
        if regex:
            try:
                compiled = re.compile(regex)
            except re.error as exc:
                raise ValueError(f"{field}: invalid regular expression: {exc}") from exc
            if "key" not in compiled.groupindex:
                raise ValueError(f"{field}: include a named key group (?P<key>...)")
    if out["match_mode"] == "regex":
        required_roles = {"image", "table"}
        required_roles.update(role for role in ("seg", "dm3") if out[f"require_{role}"])
        missing = [role for role in sorted(required_roles)
                   if not out["role_key_regex"].get(role, out["key_regex"])]
        if missing:
            raise ValueError(f"Regex matching needs key_regex or role_key_regex for: {', '.join(missing)}")
    if out["px_nm"] is not None:
        value = out["px_nm"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("px_nm must be a finite positive number or null")
        out["px_nm"] = float(value)
    if not isinstance(out["options"], dict):
        raise ValueError("options must be a JSON object")
    return out


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Read saved settings, or return defaults when the file is absent."""
    target = Path(path).expanduser()
    if not target.exists():
        return default_config()
    try:
        with target.open(encoding="utf-8-sig") as stream:
            config = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load configuration {target}: {exc}") from exc
    return validate_config(config)


def save_config(config: dict, path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    """Atomically replace only the explicitly selected config file."""
    config = validate_config(config)
    blob = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=target.parent, prefix=f".{target.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(blob)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return target


def _matches(relative: str, pattern: str) -> bool:
    relative, pattern = relative.casefold(), pattern.casefold()
    subject = relative if "/" in pattern else relative.rsplit("/", 1)[-1]
    return (fnmatch.fnmatchcase(subject, pattern)
            or (pattern.startswith("**/") and fnmatch.fnmatchcase(relative, pattern[3:])))


def _layout_match(path: Path, root: Path, rule: dict, recursive: bool):
    """Recover the measurement base using exact directory and suffix segments."""
    extension, suffix = rule["extension"], rule["suffix"]
    if path.suffix.casefold() != extension.casefold():
        return None, "extension does not match"
    relative_parent = path.parent.relative_to(root).parts
    directory_parts = PurePosixPath(rule["directory"] or "").parts
    count = len(directory_parts)
    if count and (len(relative_parent) < count or
                  tuple(part.casefold() for part in relative_parent[-count:]) !=
                  tuple(part.casefold() for part in directory_parts)):
        return None, f"parent directory must end with {rule['directory']!r}"
    base_parts = relative_parent[:-count] if count else relative_parent
    if not recursive and base_parts:
        return None, "measurement base is nested and recursive scanning is disabled"
    stem = path.name[:-len(extension)]
    if suffix and not stem.casefold().endswith(suffix.casefold()):
        return None, f"filename stem must end with literal suffix {suffix!r}"
    stripped = stem[:-len(suffix)] if suffix else stem
    if not stripped:
        return None, "removing the suffix leaves an empty measurement name"
    key = "/".join((*base_parts, stripped))
    return {"key": key, "measurement_base": "/".join(base_parts) or ".",
            "stripped_stem": stripped, "directory": rule["directory"],
            "extension": extension, "suffix": suffix}, ""


def scan_directory(config: dict) -> pd.DataFrame:
    """Return one row per key; ready rows alone are enabled for analysis.

    `issues` explains missing/duplicate roles, `*_candidates` exposes all paths,
    and `match_details` exposes the relative path and extracted key for each
    match. Files that match a role glob but fail key extraction get separate
    `key_error` rows. `frame.attrs['scan_issues']` records unreadable directories.
    """
    config = validate_config(config)
    if not config["root"].strip():
        raise ValueError("Select a root directory before scanning")
    root = Path(config["root"]).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Root directory does not exist or is not a directory: {root}")
    role_patterns = {role: _patterns(config["patterns"][role], f"patterns.{role}") for role in ROLES}
    ignored = _patterns(config["ignored_patterns"], "ignored_patterns")
    regexes = {role: re.compile(pattern) if pattern else None for role in ROLES
               for pattern in [config["role_key_regex"].get(role, config["key_regex"]) ]}
    groups: dict[str, dict] = {}
    scan_issues = []
    unmatched = []
    unmatched_details = []
    files_scanned = 0

    def get_group(key):
        identity = key.casefold() if config["match_mode"] == "layout" else key
        return groups.setdefault(identity, {"key": key, "candidates": {role: [] for role in ROLES},
                                            "details": [], "errors": []})

    def on_error(exc):
        scan_issues.append(f"Cannot scan directory: {exc}")

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=on_error):
        parent = Path(directory)
        dirs[:] = sorted((name for name in dirs
                          if not name.startswith(".") and name.casefold() not in _SKIP_DIRS
                          and not any(_matches((parent / name).relative_to(root).as_posix(), pattern)
                                      or _matches((parent / name).relative_to(root).as_posix() + "/", pattern)
                                      for pattern in ignored)), key=lambda name: (name.casefold(), name))
        if not config["recursive"]:
            if config["match_mode"] == "layout":
                role_dirs = [PurePosixPath(rule["directory"] or "").parts
                             for rule in config["roles"].values()]
                dirs[:] = [name for name in dirs
                           if any(tuple(part.casefold() for part in (parent / name).relative_to(root).parts)
                                  == tuple(part.casefold() for part in parts[:len((parent / name).relative_to(root).parts)])
                                  for parts in role_dirs)]
            else:
                dirs[:] = []
        for name in sorted(files, key=lambda value: (value.casefold(), value)):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if name.startswith((".", "~$")) or any(_matches(relative, pattern) for pattern in ignored):
                continue
            files_scanned += 1
            layout_details = {}
            if config["match_mode"] == "layout":
                matches = {}
                for role, rule in config["roles"].items():
                    detail, reason = _layout_match(path, root, rule, config["recursive"])
                    if detail is not None:
                        matches[role] = f"{rule['directory'] or '.'}/*{rule['suffix']}{rule['extension']}"
                        layout_details[role] = detail
                    elif path.suffix.casefold() == rule["extension"].casefold():
                        unmatched_details.append({"role": role, "relative_path": relative, "reason": reason})
            else:
                matches = {role: pattern for role, patterns in role_patterns.items()
                           for pattern in patterns if _matches(relative, pattern)}
            if not matches:
                unmatched.append(relative)
                continue
            for role, pattern in matches.items():
                source = path.stem if config["key_source"] == "stem" else relative
                regex = regexes[role]
                error = ""
                if config["match_mode"] == "layout":
                    key = layout_details[role]["key"]
                    source = relative
                elif config["match_mode"] == "folder":
                    key = path.parent.relative_to(root).as_posix()
                elif config["match_mode"] == "regex":
                    match = regex.search(source) if regex is not None else None
                    key = match.group("key").strip() if match and match.group("key") else ""
                    if not key:
                        error = f"{role}: key regex did not produce a nonempty key for {relative}"
                else:
                    key = path.stem
                if error:
                    key = f"[unmatched:{role}] {relative}"
                group = get_group(key)
                group["candidates"][role].append(str(path))
                group["details"].append({"role": role, "relative_path": relative, "source": source,
                                         "key": "" if error else key, "pattern": pattern,
                                         "key_regex": regex.pattern if regex is not None else "",
                                         **layout_details.get(role, {})})
                if error:
                    group["errors"].append(error)
                if len(matches) > 1:
                    group["errors"].append(f"File matches multiple roles {', '.join(matches)}: {relative}")

    rows = []
    for _, group in sorted(groups.items(), key=lambda item: (item[0].casefold(), item[0])):
        key = group["key"]
        errors = list(dict.fromkeys(group["errors"]))
        warnings = []
        row = {"image_id": key, "px_nm": config["px_nm"], "match_details": group["details"]}
        for role in ROLES:
            candidates = sorted(set(group["candidates"][role]), key=lambda value: (value.casefold(), value))
            row[f"{role}_candidates"] = candidates
            row[f"{role}_path"] = candidates[0] if len(candidates) == 1 else ""
            required = role in {"image", "table"} or config[f"require_{role}"]
            if not candidates:
                (errors if required else warnings).append(f"Missing {'required' if required else 'optional'} {role} file")
            elif len(candidates) > 1:
                errors.append(f"Ambiguous {role}: {len(candidates)} candidates; adjust rules or choose manually")
        row["status"] = "key_error" if key.startswith("[unmatched:") else "error" if errors else "ready"
        row["enabled"] = not errors
        row["issues"] = "; ".join(errors + warnings)
        rows.append(row)
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    frame.attrs.update(root=str(root), files_scanned=files_scanned,
                       unmatched_files=unmatched, unmatched_details=unmatched_details,
                       scan_issues=scan_issues)
    return frame
