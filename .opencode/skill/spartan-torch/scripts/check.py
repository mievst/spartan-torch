#!/usr/bin/env python3
"""
spartan-check: Quick local check (subset of review).
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent


def run(cmd: list[str], description: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"> {description}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"FAILED: {description}")
        return False
    print(f"PASSED: {description}")
    return True


def main() -> int:
    all_passed = True

    all_passed &= run(["uv", "run", "ruff", "check", "."], "Ruff lint")
    all_passed &= run(["uv", "run", "ruff", "format", "--check", "."], "Ruff format")
    all_passed &= run(["uv", "run", "ty", "check", "src/spartan_torch"], "Ty type check")
    all_passed &= run(["uv", "run", "pytest", "tests/unit", "-x", "--tb=short"], "Unit tests")

    print("\n" + "=" * 60)
    if all_passed:
        print("QUICK CHECK PASSED")
        return 0
    else:
        print("QUICK CHECK FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
