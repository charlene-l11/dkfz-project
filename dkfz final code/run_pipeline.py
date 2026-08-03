from __future__ import annotations

import sys
from pathlib import Path

from median_pipeline.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        scenario_configs = sorted([
            *Path.cwd().glob("*.yaml"),
            *Path.cwd().glob("*.yml"),
        ])
        if len(scenario_configs) == 1:
            sys.argv.extend(["run", "--config", str(scenario_configs[0])])
        else:
            raise SystemExit(
                "Use: python run_pipeline.py --config <scenario>/<file>.yaml"
            )
    elif sys.argv[1].startswith("-"):
        sys.argv.insert(1, "run")
    main()
