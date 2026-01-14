"""Update spirosim/_version.py using GitVersion locally."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _ensure_repo_on_path(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _run_gitversion(repo_root: Path) -> dict[str, str] | None:
    commands = []
    if shutil.which("gitversion"):
        commands.append(["gitversion"])
    if shutil.which("dotnet"):
        commands.append(
            [
                "dotnet",
                "tool",
                "run",
                "dotnet-gitversion",
                "--tool-manifest",
                str(repo_root / ".config" / "dotnet-tools.json"),
                "--",
            ]
        )

    for command in commands:
        try:
            result = subprocess.run(
                command + ["/output", "json", "/nofetch"],
                check=True,
                capture_output=True,
                text=True,
                cwd=repo_root,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            continue
    return None


def _read_version_fields(metadata: dict[str, str]) -> tuple[str, str]:
    version = metadata.get("FullSemVer") or metadata.get("fullSemVer") or "0.0.0-dev"
    sha = metadata.get("ShortSha") or metadata.get("shortSha") or "unknown"
    return version, sha


def main() -> int:
    try:
        repo_root = _repo_root()
    except subprocess.CalledProcessError:
        print("Unable to locate git repository root.", file=sys.stderr)
        return 1

    _ensure_repo_on_path(repo_root)
    from spirosim.versioning import write_version_file

    metadata = _run_gitversion(repo_root)
    if not metadata:
        print(
            "GitVersion not available. Install it or run `dotnet tool restore`.",
            file=sys.stderr,
        )
        return 0

    version, sha = _read_version_fields(metadata)
    version_path = repo_root / "spirosim" / "_version.py"
    write_version_file(version_path, version, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
