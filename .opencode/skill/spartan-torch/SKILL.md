# spartan-torch Opencode Skill

Custom commands for spartan-torch development workflow.

## Installation

This skill is automatically available when working in the spartan-torch repository.

## Commands

### `spartan-review` - Full Strict Review Pipeline
Run all quality gates locally before pushing.

```bash
opencode run spartan-review
```

**Runs:**
- `ruff check .` + `ruff format --check .`
- `ty check src/spartan_torch`
- `pytest tests/unit -x --tb=short`
- `bandit -r src/spartan_torch`

**Policy:** Zero tolerance. Any warning = failure.

---

### `spartan-check` - Quick Check
Fast subset for rapid feedback during development.

```bash
opencode run spartan-check
```

**Runs:** ruff, ty, unit tests only. ~30 seconds.

---

### `spartan-new-block` - Scaffold New Block
Create new block with tests and VRAM annotations.

```bash
opencode run spartan-new-block --category attention --name MyAttention
```

**Creates:**
- `src/spartan_torch/blocks/attention/my_attention.py` (template with VRAM annotation)
- `tests/unit/blocks/attention/test_my_attention.py` (shape + gradcheck)
- Updates `__init__.py` exports

---

### `spartan-release` - Release Automation
Version bump, build, and tag.

```bash
# Patch release (0.1.0 -> 0.1.1)
opencode run spartan-release --patch

# Minor release (0.1.0 -> 0.2.0)
opencode run spartan-release --minor

# Major release (0.1.0 -> 1.0.0)
opencode run spartan-release --major

# Dry run (no tag, no publish)
opencode run spartan-release --patch --dry-run
```

**Does:**
1. Runs all quality gates (ruff, ty, pytest)
2. Bumps version in pyproject.toml
3. Builds wheel + sdist
4. Git commit + tag + push

---

## Scripts Location

All scripts are in `.opencode/skill/spartan-torch/scripts/`:
- `review.py` - Full review pipeline
- `check.py` - Quick check
- `new_block.py` - Block scaffolder
- `release.py` - Release automation

## Or via make

```bash
make check
make lint
make format
make typecheck
make test
make security
make build
```
