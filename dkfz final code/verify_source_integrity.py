from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    baseline_path = project_root / "source_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures = []
    for source in baseline["sources"]:
        source_root = Path(source["root"])
        for item in source["files"]:
            path = source_root / item["relative_path"]
            if not path.is_file():
                failures.append(f"missing: {path}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
            if digest != item["sha256"]:
                failures.append(f"changed: {path}")
    if failures:
        raise SystemExit("Source-integrity check failed:\n" + "\n".join(failures))
    print("Both immutable HTC source trees match source_baseline.json")


if __name__ == "__main__":
    main()
