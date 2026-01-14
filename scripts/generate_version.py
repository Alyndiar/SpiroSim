"""Generate spirosim version metadata for CI and local builds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from spirosim.versioning import resolve_version, write_version_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spirosim") / "_version.py",
        help="Path to the version metadata file.",
    )
    parser.add_argument(
        "--version",
        dest="version",
        help="Override SEMVER value (defaults to SEMVER env var).",
    )
    parser.add_argument(
        "--sha",
        dest="sha",
        help="Override SHA value (defaults to SHA env var).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = dict(os.environ)
    if args.version:
        env["SEMVER"] = args.version
    if args.sha:
        env["SHA"] = args.sha
    resolved = resolve_version(env)
    version = resolved.version
    sha = resolved.sha
    write_version_file(args.output, version, sha)


if __name__ == "__main__":
    main()
