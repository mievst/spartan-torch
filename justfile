# spartan-torch task runner (just)
# NOTE: don't run `just sync` / `just clean` while Jupyter kernels are alive —
# they lock .pyd files and uv can't finish the sync.

set shell := ["sh", "-cu"]
set windows-shell := ["pwsh", "-NoProfile", "-Command"]

venv_python := if os_family() == "windows" {
    ".venv\\Scripts\\python.exe"
} else {
    ".venv/bin/python"
}

# Run the test suite. Uses the venv python directly, never touches the
# environment (no uv sync, so safe even if the env is out of date).
test:
    {{venv_python}} -m pytest

# Install / reconcile the virtual environment from the lockfile.
sync:
    uv sync --extra dev --extra experiments

# Delete the virtual environment.
clean:
    {{ if os_family() == "windows" {
        "if (Test-Path .venv) { Remove-Item -Recurse -Force .venv; Write-Host 'Removed .venv' } else { Write-Host 'No .venv to remove' }"
    } else {
        "rm -rf .venv && echo 'Removed .venv'"
    } }}
