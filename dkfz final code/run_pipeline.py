from __future__ import annotations

import sys

from median_pipeline.cli import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["run", "--config", "configs/manifold_settings.yaml"])
    main()
