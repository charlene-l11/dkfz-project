#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import math
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = [
    "spectrum_fromCSV1_(500.0-995.0)_masked_data_0_derivative.csv",
    "spectrum_fromCSV1_(500.0-995.0)*_masked_data_0_derivative.csv",
 ]

DEFAULT_TRAINING_SETTINGS = {
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "seed": 0,
    "max_epochs": 10,
    "batch_size": None,
    "learning_rate": None,
    "class_weight_method": "none",
    "class_weights": None,
    "oversampling": False,
    "standardize": False,
    "accelerator": "auto",
}


def unique_values(values) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(column for row in rows for column in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_run_output_dir(base_dir: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%d%m%Y_%H%M")
    output_dir = base_dir / f"{timestamp}_prepCSV"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    result = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        result.append(ch)
    return "".join(result).rstrip()


def parse_scalar(value: str) -> Any:
    value = strip_inline_comment(value).strip()
    if value == "":
        return None
    if value in {"True", "true"}:
        return True
    if value in {"False", "false"}:
        return False
    if value in {"None", "null", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if re.match(r"^-?\d+$", value):
            return int(value)
        if re.match(r"^-?\d+\.\d+$", value):
            return float(value)
    except Exception:
        pass
    return value


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    """
    Small YAML parser for the DKFZ-style settings files used here.
    It supports:
      key: value
      section:
        subkey: value
      list_key:
        - item
    It intentionally does not need PyYAML.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    root: dict[str, Any] = {}
    current_section: str | None = None
    section_is_list: dict[str, bool] = {}

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        line = strip_inline_comment(raw_line)
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0:
            current_section = None
            if stripped.endswith(":"):
                key = stripped[:-1].strip()
                root[key] = {}
                current_section = key
                section_is_list[key] = False
            elif ":" in stripped:
                key, value = stripped.split(":", 1)
                root[key.strip()] = parse_scalar(value)
            continue

        if current_section is None:
            continue

        if stripped.startswith("- "):
            if not section_is_list.get(current_section, False):
                root[current_section] = []
                section_is_list[current_section] = True
            root[current_section].append(parse_scalar(stripped[2:]))
            continue

        if ":" in stripped:
            if section_is_list.get(current_section, False):
                continue
            key, value = stripped.split(":", 1)
            root[current_section][key.strip()] = parse_scalar(value)

    return root


def parse_python_settings(path: Path) -> dict[str, Any]:
    """
    Reads a Python path-overview file without executing it.
    It supports assignments like:
      label_label = "_labelling_001.txt"
      organlist = [...]
      train_ratio = 0.7
      training = {"max_epochs": 30, "class_weight_method": "inverse"}
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(text)

    data: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if not names:
            continue

        try:
            value = ast.literal_eval(node.value)
        except Exception:
            continue

        for name in names:
            data[name] = value

    # Keep inline comments next to organlist entries because they explain the biological condition.
    comment_by_path: dict[str, str] = {}
    for line in text.splitlines():
        match = re.search(r'["\']([^"\']+)["\']\s*,?\s*#\s*(.+)$', line)
        if match:
            comment_by_path[match.group(1)] = match.group(2).strip()
    data["_organ_comments"] = comment_by_path

    return data


def load_settings_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return parse_simple_yaml(path)
    if suffix == ".py":
        return parse_python_settings(path)
    raise ValueError(f"Unsupported settings file type: {path}. Use .yaml, .yml, or .py")


def settings_block(settings: dict[str, Any]) -> dict[str, Any]:
    """
    Training/model parameters may be top-level or inside one of these blocks.

    Recommended YAML style:
      cat_pig_model:
        output_dir: "C:/..."
        train_ratio: 0.7
        val_ratio: 0.15
        test_ratio: 0.15
        seed: 42
        max_epochs: 30
        batch_size: 32
        learning_rate: 0.0001
        class_weight_method: inverse
        standardize: true
        oversampling: false
    """
    for key in ("cat_pig_model", "cat_pig_training", "model_training", "training", "ml_training"):
        value = settings.get(key)
        if isinstance(value, dict):
            return value
    return {}


def get_setting(settings: dict[str, Any], name: str, default: Any = None) -> Any:
    block = settings_block(settings)
    if name in block:
        return block[name]
    if name in settings:
        return settings[name]
    return default


def collect_training_settings(settings: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = dict(DEFAULT_TRAINING_SETTINGS)

    # Read values from the YAML/Python settings file first.
    for key in training:
        value = get_setting(settings, key, None)
        if value is not None:
            training[key] = value

    # Then CLI values override the file.
    cli_map = {
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "class_weight_method": args.class_weight_method,
        "class_weights": args.class_weights,
        "accelerator": args.accelerator,
    }
    for key, value in cli_map.items():
        if value is not None:
            training[key] = value

    if args.standardize:
        training["standardize"] = True
    if args.oversampling:
        training["oversampling"] = True

    training["train_ratio"] = float(training["train_ratio"])
    training["val_ratio"] = float(training["val_ratio"])
    training["test_ratio"] = float(training["test_ratio"])
    training["seed"] = int(training["seed"])
    training["max_epochs"] = int(training["max_epochs"])
    training["standardize"] = coerce_bool(training["standardize"])
    training["oversampling"] = coerce_bool(training["oversampling"])

    if training["batch_size"] is not None:
        training["batch_size"] = int(training["batch_size"])
    if training["learning_rate"] is not None:
        training["learning_rate"] = float(training["learning_rate"])

    return training


def apply_prefix_maps(path_text: str, prefix_maps: list[str] | None) -> str:
    result = str(path_text).replace("\\", "/")
    for mapping in prefix_maps or []:
        if "=" not in mapping:
            raise ValueError(f"Invalid --path-prefix-map value {mapping!r}. Use OLD=NEW.")
        old, new = mapping.split("=", 1)
        old = old.replace("\\", "/").rstrip("/")
        new = new.replace("\\", "/").rstrip("/")
        if result.startswith(old):
            result = new + result[len(old):]
    return result


def label_from_folder(path: Path) -> str:
    parts = list(path.parts)
    candidates = [p for p in parts if re.match(r"^Cat_\d+_.+", p)]
    if candidates:
        return re.sub(r"^Cat_\d+_", "", candidates[-1])
    if path.name.lower() == "data" and path.parent.name:
        return re.sub(r"^Cat_\d+_", "", path.parent.name)
    return path.name


def parse_source(value: str) -> dict[str, Any]:
    if "=" in value:
        label, path = value.split("=", 1)
        if not label.strip():
            raise argparse.ArgumentTypeError("Source label cannot be empty.")
        return {
            "label": label.strip(),
            "source_dir": str(Path(path).expanduser()),
            "label_file_pattern": "",
            "source_key": label.strip(),
            "source_comment": "",
            "hypergui": "",
        }

    path = Path(value).expanduser()
    label = label_from_folder(path)
    return {
        "label": label,
        "source_dir": str(path),
        "label_file_pattern": "",
        "source_key": label,
        "source_comment": "",
        "hypergui": "",
    }


def sources_from_settings(settings: dict[str, Any], prefix_maps: list[str] | None) -> list[dict[str, Any]]:
    """
    Accepts either:
      Python style:
        label_label = "_labelling_001.txt"
        organlist = ["Z:/.../Cat_0001_stomach/data", ...]

      YAML style:
        experiment_folders:
          1_stom_rat: "Z:/.../Cat_0001_stomach"
        labelling_file:
          1_stom_rat: "_labelling_001.txt"
        hyperguis:
          1_stom_rat: "_hypergui_1"

    Optional label override:
      labels:
        1_stom_rat: stomach
      label_names:
        1_stom_rat: stomach
      class_names:
        1_stom_rat: stomach
    """
    sources: list[dict[str, Any]] = []

    # Python path overview format
    organlist = settings.get("organlist")
    if isinstance(organlist, list):
        label_pattern = str(settings.get("label_label", ""))
        comments = settings.get("_organ_comments", {}) if isinstance(settings.get("_organ_comments"), dict) else {}
        for raw in organlist:
            mapped = apply_prefix_maps(str(raw), prefix_maps)
            folder = Path(mapped)
            sources.append({
                "label": label_from_folder(folder),
                "source_dir": str(folder),
                "label_file_pattern": label_pattern,
                "source_key": label_from_folder(folder),
                "source_comment": comments.get(str(raw), ""),
                "hypergui": "",
            })
        return sources

    # YAML manifold/settings format
    experiment_folders = settings.get("experiment_folders", {})
    labelling_file = settings.get("labelling_file", {})
    hyperguis = settings.get("hyperguis", {})

    label_overrides = {}
    for key in ("labels", "label_names", "class_names"):
        if isinstance(settings.get(key), dict):
            label_overrides.update(settings[key])

    if not isinstance(experiment_folders, dict):
        return []

    for key, raw_folder in experiment_folders.items():
        mapped = apply_prefix_maps(str(raw_folder), prefix_maps)
        folder = Path(mapped)
        data_folder = folder / "data"
        source_dir = data_folder if data_folder.exists() else folder

        sources.append({
            "label": str(label_overrides.get(key, label_from_folder(folder))),
            "source_dir": str(source_dir),
            "label_file_pattern": labelling_file.get(key, "") if isinstance(labelling_file, dict) else "",
            "hypergui": hyperguis.get(key, "") if isinstance(hyperguis, dict) else "",
            "source_key": key,
            "source_comment": "",
        })

    return sources


def subject_from_path(path: Path) -> str:
    for part in path.parts:
        if re.match(r"^[PM]\d+", part):
            return part.split("_")[0]
    for part in path.parts:
        if re.match(r"^Cat_\d+_", part):
            return part
    return path.parent.name


def timestamp_from_path(path: Path) -> str:
    return next(
        (part for part in path.parts if re.match(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$", part)),
        path.parent.name,
    )


def image_name_from_file(path: Path) -> str:
    subject = subject_from_path(path)
    timestamp = timestamp_from_path(path)
    stem = path.stem.replace("spectrum_from", "")
    return f"{subject}#{timestamp}#{stem}"


def rows_from_sources(sources: list[dict[str, Any]], patterns: list[str]) -> list[dict]:
    rows = []

    for source in sources:
        label = str(source["label"])
        root = Path(str(source["source_dir"])).expanduser()

        if not root.exists():
            raise FileNotFoundError(
                f"Source folder does not exist: {root}\n"
                "If this is a Z:/ network path, make sure the drive is mounted or use --path-prefix-map OLD=NEW."
            )

        paths = sorted({path for pattern in patterns for path in root.rglob(pattern) if path.is_file()})
        if not paths:
            raise FileNotFoundError(f"No spectrum CSV files matching {patterns} below {root}")

        for path in paths:
            rows.append({
                "file_path": str(path.resolve()),
                "label": label,
                "subject_name": subject_from_path(path),
                "timestamp": timestamp_from_path(path),
                "image_name": image_name_from_file(path),
                "annotation_name": "csv_spectrum",
                "source_dir": str(root),
                "source_key": source.get("source_key", label),
                "source_comment": source.get("source_comment", ""),
                "label_file_pattern": source.get("label_file_pattern", ""),
                "hypergui": source.get("hypergui", ""),
            })

    labels_by_path: dict[str, set[str]] = {}
    for row in rows:
        labels_by_path.setdefault(row["file_path"], set()).add(row["label"])

    conflicts = {path: labels for path, labels in labels_by_path.items() if len(labels) > 1}
    if conflicts:
        preview = "\n".join(f"{path}: {sorted(labels)}" for path, labels in list(conflicts.items())[:10])
        raise ValueError(
            f"{len(conflicts)} files were assigned to more than one label. "
            f"Source directories must not overlap.\n{preview}"
        )

    seen = set()
    unique_rows = []
    for row in rows:
        if row["file_path"] in seen:
            continue
        seen.add(row["file_path"])
        unique_rows.append(row)

    return unique_rows


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    required = {"label", "subject_name"}
    missing = required - columns
    if missing:
        raise ValueError(f"Input manifest is missing columns: {sorted(missing)}")

    feature_columns = [column for column in columns if column.startswith("feature_")]
    if "file_path" not in columns and not feature_columns:
        raise ValueError("Input manifest needs file_path or feature_* columns.")

    for index, row in enumerate(rows):
        if "image_name" not in columns or not row.get("image_name"):
            row["image_name"] = f"{row['subject_name']}#{index}"
        if "annotation_name" not in columns or not row.get("annotation_name"):
            row["annotation_name"] = "csv_spectrum"

    if "file_path" in columns:
        for row in rows:
            row["file_path"] = str(Path(row["file_path"]).expanduser().resolve())

        missing_files = [row["file_path"] for row in rows if not Path(row["file_path"]).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"Manifest references {len(missing_files)} missing files; first: {missing_files[0]}"
            )

        seen = set()
        unique_rows = []
        for row in rows:
            if row["file_path"] in seen:
                continue
            seen.add(row["file_path"])
            unique_rows.append(row)
        rows = unique_rows

    return rows


def split_score(rows: list[dict], assignment: dict[str, str], labels: list[str], ratios: dict[str, float]) -> float:
    score = 0.0
    for name, ratio in ratios.items():
        subset = [row for row in rows if assignment[str(row["subject_name"])] == name]
        missing_labels = len(set(labels) - {str(row["label"]) for row in subset})
        score += missing_labels * 1000
        score += abs(len(subset) / len(rows) - ratio)
    return score


def grouped_split(rows: list[dict], ratios: dict[str, float], seed: int) -> list[str]:
    subjects = sorted({str(row["subject_name"]) for row in rows})

    if len(subjects) == 1:
        print("Warning: only one subject/group detected. All rows will be placed in train.")
        return ["train" for _ in rows]

    if len(subjects) == 2:
        assignment = {subjects[0]: "train", subjects[1]: "val"}
        return [assignment[str(row["subject_name"])] for row in rows]

    labels = unique_values(row["label"] for row in rows)
    label_subjects: dict[str, set[str]] = {}
    for row in rows:
        label_subjects.setdefault(str(row["label"]), set()).add(str(row["subject_name"]))

    insufficient = {label: len(subjects) for label, subjects in label_subjects.items() if len(subjects) < 3}
    if insufficient:
        print(
            "Warning: these labels occur in fewer than three subjects and cannot appear in every split: "
            + ", ".join(f"{label} ({count})" for label, count in insufficient.items())
        )

    rng = random.Random(seed)
    best_assignment = None
    best_score = float("inf")

    min_groups_per_split = min(len(labels), len(subjects) // 3)
    n_val = max(1, int(round(len(subjects) * ratios["val"])), min_groups_per_split)
    n_test = max(1, int(round(len(subjects) * ratios["test"])), min_groups_per_split)
    n_train = len(subjects) - n_val - n_test

    if n_train < min_groups_per_split:
        n_train = min_groups_per_split
        remaining = len(subjects) - n_train
        n_val = max(1, remaining // 2)
        n_test = max(1, remaining - n_val)

    for _ in range(10000):
        shuffled = list(subjects)
        rng.shuffle(shuffled)
        assignment = {
            **{subject: "train" for subject in shuffled[:n_train]},
            **{subject: "val" for subject in shuffled[n_train:n_train + n_val]},
            **{subject: "test" for subject in shuffled[n_train + n_val:n_train + n_val + n_test]},
        }

        score = split_score(rows, assignment, labels, ratios)
        if score < best_score:
            best_assignment, best_score = assignment, score

    if best_assignment is None:
        raise RuntimeError("Could not create a split assignment.")

    return [best_assignment[str(row["subject_name"])] for row in rows]


def resolve_output_dir(settings: dict[str, Any], args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    output_dir = get_setting(settings, "output_dir", None)
    if output_dir:
        return Path(str(output_dir))
    return Path.cwd() / "training_runs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/val/test CSVs from spectrum folders, a YAML settings file, a Python path overview file, or a manifest."
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--config-file", type=Path, help="YAML or Python settings/path-overview file.")
    source_group.add_argument("--settings-yaml", type=Path, help="Alias for --config-file with a YAML file.")
    source_group.add_argument("--path-overview", type=Path, help="Alias for --config-file with a Python path overview file.")
    source_group.add_argument("--source", action="append", type=parse_source, help="Manual source as LABEL=PATH. Repeat for every class.")
    source_group.add_argument("--input-manifest", type=Path, help="CSV with file_path, label, and subject_name.")

    parser.add_argument("--pattern", action="append", help="Filename/glob to scan. Repeat as needed.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--class-weight-method", choices=["none", "inverse", "balanced", "softmin", "nll"])
    parser.add_argument("--class-weights")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"])
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--oversampling", action="store_true")
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        help='Optional path conversion as OLD=NEW, e.g. "Z:/TIVITA_Cat=C:/Users/c758g/Documents/Cat_Pig". Repeat if needed.',
    )

    args = parser.parse_args()

    config_path = args.config_file or args.settings_yaml or args.path_overview

    settings: dict[str, Any] = {}
    if config_path:
        settings = load_settings_file(config_path)

    training = collect_training_settings(settings, args)
    ratios = {
        "train": training["train_ratio"],
        "val": training["val_ratio"],
        "test": training["test_ratio"],
    }
    if not math.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    patterns = args.pattern or get_setting(settings, "patterns", None) or get_setting(settings, "spectrum_patterns", None) or DEFAULT_PATTERNS
    if isinstance(patterns, str):
        patterns = [patterns]

    if args.input_manifest:
        rows = load_manifest(args.input_manifest)
        sources = []
    elif args.source:
        sources = args.source
        rows = rows_from_sources(sources, patterns)
    else:
        sources = sources_from_settings(settings, args.path_prefix_map)
        if not sources:
            raise ValueError(
                "No input sources found. The config file needs either organlist or experiment_folders, "
                "or use --source LABEL=PATH."
            )

        print("Detected source folders/classes:")
        for source in sources:
            comment = f" ({source.get('source_comment')})" if source.get("source_comment") else ""
            print(f"  {source['label']}: {source['source_dir']}{comment}")

        rows = rows_from_sources(sources, patterns)

    labels = unique_values(row["label"] for row in rows)
    label_mapping = {label: index for index, label in enumerate(labels)}
    splits = grouped_split(rows, ratios, training["seed"])

    for row, split in zip(rows, splits):
        row["label"] = str(row["label"])
        row["label_index"] = label_mapping[row["label"]]
        row["split"] = split

    output_dir = build_run_output_dir(resolve_output_dir(settings, args))

    fieldnames = list(dict.fromkeys(column for row in rows for column in row))
    write_csv(output_dir / "manifest.csv", rows, fieldnames=fieldnames)

    (output_dir / "labels.json").write_text(
        json.dumps({"labels": labels, "mapping": label_mapping}, indent=2),
        encoding="utf-8",
    )

    feature_columns = [column for row in rows for column in row if column.startswith("feature_")]
    feature_columns = list(dict.fromkeys(feature_columns))

    pipeline_config = {
        "settings_file": str(config_path) if config_path else None,
        "output_dir": str(output_dir),
        "data_dir": str(output_dir),
        "patterns": patterns,
        "training": training,
        "sources": sources,
        "labels": labels,
        "label_mapping": label_mapping,
    }

    (output_dir / "dataset.json").write_text(
        json.dumps({"feature_columns": feature_columns, "input_type": "spectrum_csv", "patterns": patterns}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "pipeline_config.json").write_text(json.dumps(pipeline_config, indent=2), encoding="utf-8")

    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        write_csv(output_dir / f"{split}.csv", subset, fieldnames=fieldnames)
        counts = {label: sum(1 for row in subset if row["label"] == label) for label in labels}
        subjects = {row["subject_name"] for row in subset}
        print(f"{split}: {len(subset)} spectra, {len(subjects)} subjects, labels={counts}")

    print(f"Prepared data folder: {output_dir}")
    print(f"Pipeline config: {output_dir / 'pipeline_config.json'}")


if __name__ == "__main__":
    main()
