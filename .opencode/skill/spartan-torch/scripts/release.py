#!/usr/bin/env python3
"""
spartan-release: Version bump, build, and tag.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
PYPROJECT = ROOT / "pyproject.toml"


def run(cmd: list[str], description: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"> {description}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode == 0


def get_version() -> str:
    content = PYPROJECT.read_text()
    match = re.search(r'^version\s*=\s*"(.+)"', content, re.MULTILINE)
    if not match:
        raise RuntimeError("Cannot find version in pyproject.toml")
    return match.group(1)


def bump_version(version: str, bump: str) -> str:
    parts = version.split(".")
    if len(parts) != 3:
        raise RuntimeError(f"Invalid version format: {version}")
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    if bump == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump == "minor":
        minor += 1
        patch = 0
    else:
        patch += 1
    return f"{major}.{minor}.{patch}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Release spartan-torch")
    parser.add_argument("--patch", action="store_true", help="Bump patch version")
    parser.add_argument("--minor", action="store_true", help="Bump minor version")
    parser.add_argument("--major", action="store_true", help="Bump major version")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no git tag)")
    args = parser.parse_args()

    if not (args.patch or args.minor or args.major):
        parser.error("Must specify --patch, --minor, or --major")

    bump = "patch" if args.patch else ("minor" if args.minor else "major")
    old_version = get_version()
    new_version = bump_version(old_version, bump)

    print(f"Releasing: {old_version} -> {new_version} ({bump})")

    # Pre-release checks
    if not run(["uv", "run", "ruff", "check", "."], "Ruff lint"):
        return 1
    if not run(["uv", "run", "ruff", "format", "--check", "."], "Ruff format"):
        return 1
    if not run(["uv", "run", "ty", "check", "src/spartan_torch"], "Ty type check"):
        return 1
    if not run(["uv", "run", "pytest", "tests/unit", "-x", "--tb=short"], "Unit tests"):
        return 1

    # Bump version in pyproject.toml
    content = PYPROJECT.read_text()
    content = content.replace(f'version = "{old_version}"', f'version = "{new_version}"')
    PYPROJECT.write_text(content)
    print(f"Bumped version to {new_version}")

    # Build
    if not run(["uv", "build", "--wheel", "--sdist"], "Build package"):
        return 1

    if args.dry_run:
        # Revert version bump
        content = PYPROJECT.read_text()
        content = content.replace(f'version = "{new_version}"', f'version = "{old_version}"')
        PYPROJECT.write_text(content)
        print("\nDRY RUN - version reverted, not tagging")
        return 0

    # Git commit + tag
    if not run(["git", "add", "pyproject.toml"], "Git add"):
        return 1
    if not run(["git", "commit", "-m", f"chore: release v{new_version}"], "Git commit"):
        return 1
    if not run(["git", "tag", f"v{new_version}"], "Git tag"):
        return 1
    if not run(["git", "push", "origin", "main", "--tags"], "Git push"):
        return 1

    print(f"\nRelease v{new_version} pushed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
