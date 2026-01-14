"""Configure git to use the repository hooks."""

from __future__ import annotations

import subprocess


def main() -> None:
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], check=True)
    print("Configured core.hooksPath to .githooks")


if __name__ == "__main__":
    main()
